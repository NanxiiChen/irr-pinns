import datetime
import sys
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap, jit
import orbax.checkpoint as ocp

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from examples.wave import (
    PINN,
    WaveSampler,
    evaluate1D,
    cfg,
)
from pinn import (
    CausalWeightor,
    MetricsTracker,
    train_step,
    create_train_state,
)


class WavePINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ref_sol_bc(self, x, t):
        # left boundary: p(x=0, t) = A*sin(omega*t)
        A = self.cfg.AMPLITUDE
        omega = self.cfg.OMEGA
        Tc = self.cfg.Tc
        return A * jnp.sin(omega * t * Tc)
    
    def net_bc_right(self, params, x, t):
        Tc = self.cfg.Tc
        Lc = self.cfg.Lc
        c = self.cfg.VELOCITY
        du_dt = jax.jacrev(self.net_u, argnums=2)(
            params, x, t
        )[0] / Tc
        du_dx = jax.jacrev(self.net_u, argnums=1)(
            params, x, t
        )[0] / Lc
        res = du_dt + c * du_dx
        return res.squeeze()
    
    def loss_bc_left(self, params, batch, *args, **kwargs):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = self.ref_sol_bc(x, t)
        return jnp.mean((u - ref) ** 2), {}
    
    def loss_bc_right(self, params, batch, *args, **kwargs):
        x, t = batch
        res = vmap(self.net_bc_right, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(res ** 2), {}
    
    def loss_ic(self, params, batch, *args, **kwargs):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(u ** 2), {}
        

causal_weightor = CausalWeightor(cfg.CAUSAL_CONFIGS["chunks"], cfg.DOMAIN[-1])
loss_terms = [
    "pde",
    "ic",
    "bc_left",
    "bc_right",
    "irr",
]

pinn = WavePINN(
    config=cfg, causal_weightor=causal_weightor, loss_terms=loss_terms)

init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model,
    model_key,
    cfg.LR,
    decay=cfg.DECAY,
    decay_every=cfg.DECAY_EVERY,
    xdim=len(cfg.DOMAIN) - 1,
    end_value=1e-5,
    opt_method=cfg.OPTIMIZER
)

if cfg.RESUME is not None:
    ckpt = ocp.StandardCheckpointer()
    restore_state = ckpt.restore(cfg.RESUME)
    # state load the params from the checkpoint
    state = state.replace(
        params=restore_state["params"],
    )
    print(f"Restored from {cfg.RESUME}")

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
ckpt = ocp.StandardCheckpointer()
sampler = WaveSampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)

start_time = time.time()
for epoch in range(cfg.EPOCHS):

    if epoch % 10 == 0:
        batch = sampler.sample(
            fns=[pinn.net_pde,],
            params=state.params,
        )

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        pinn.loss_fn,
        state,
        batch,
        cfg.CAUSAL_CONFIGS["eps"],
    )

    if cfg.CAUSAL_WEIGHT:
        new_eps = causal_weightor.update_causal_eps(
            cfg.CAUSAL_CONFIGS["eps"],
            aux_vars["causal_weights"],
            cfg.CAUSAL_CONFIGS,
        )
        cfg.CAUSAL_CONFIGS.update({"eps": new_eps})

    if epoch % cfg.SAVE_EVERY == 0:
        if epoch % (cfg.SAVE_EVERY * 10) == 0:
            ckpt.save(log_path + f"/model-{epoch}", state)

        fig, error = evaluate1D(
            pinn,
            state.params,
            cfg.DATA_PATH,
            xlim=cfg.DOMAIN[0],
        )
        metrics_tracker.register_figure(epoch, fig, "fisher-error")
        plt.close(fig)

        print(f"Epoch {epoch}, Loss: {weighted_loss:.4e}, error: {error:.4e},")
        metrics_tracker.register_scalars(
            epoch,
            names=["loss/weighted"]
            + [f"loss/{term}" for term in loss_terms]
            + [f"weight/{term}" for term in loss_terms]
            + ["error"],
            values=[
                weighted_loss,
                *loss_components,
                *weight_components,
                error,
            ]
        )

        if cfg.CAUSAL_WEIGHT and epoch % cfg.SAVE_EVERY == 0:
            fig = pinn.causal_weightor.plot_causal_info(
                aux_vars["causal_weights"],
                aux_vars["loss_chunks"],
                cfg.CAUSAL_CONFIGS["eps"],
            )
            fig.suptitle(f"eps: {cfg.CAUSAL_CONFIGS['eps']:.2e}")
            metrics_tracker.register_figure(epoch, fig, "causal_info")
            plt.close(fig)

        metrics_tracker.flush()


end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
