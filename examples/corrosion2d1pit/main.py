"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""

import datetime
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.training import train_state
from jax import jit, random, vmap

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from pf_pinn import *
from examples.corrosion2d1pit.configs import Config as cfg


# from jax import config
# config.update("jax_disable_jit", True)


class Sampler:

    def __init__(
        self,
        n_samples,
        domain=((-0.5, 0.5), (0, 0.5), (0, 1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
            "model": None,
            "state": None,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def adaptive_sampling(self, residual_fn):
        key, self.key = random.split(self.key)
        adaptive_base = lhs_sampling(
            self.mins,
            self.maxs,
            self.adaptive_kw["num"] * self.adaptive_kw["ratio"],
            key=key,
        )
        residuals = residual_fn(adaptive_base)
        max_residuals, indices = jax.lax.top_k(
            jnp.abs(residuals), self.adaptive_kw["num"]
        )
        return adaptive_base[indices]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples * 2],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_pde_rar(self, pde_name="ac"):
        key, self.key = random.split(self.key)
        batch = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples*2, self.n_samples, self.n_samples * 3],
            key,
        )

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["params"]
            x, t = batch[:, :-1], batch[:, -1:]
            fn = model.net_ac if pde_name == "ac" else model.net_ch
            return vmap(fn, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples ** 2,
            key=key
        )
        x_local = lhs_sampling(
            mins=[-0.15, 0], maxs=[0.15, 0.15], num=self.n_samples**2 * 5, key=self.key
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)
    
        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key
        )
        top = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][1], x1t[:, 1:2]],
            axis=1,
        )
        x2t = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[1][1], self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key
        )
        left = jnp.concatenate(
            [jnp.ones_like(x2t[:, 0:1]) * self.domain[0][0], x2t[:, 0:1], x2t[:, 1:2]],
            axis=1,
        )
        right = jnp.concatenate(
            [jnp.ones_like(x2t[:, 0:1]) * self.domain[0][1], x2t[:, 0:1], x2t[:, 1:2]],
            axis=1,
        )

        # local: x1 \in (self.domain[0][0]/20, self.domain[0][1]/20), x2 = self.domain[1][0], t \in (self.domain[2][0] + self.domain[2][1] / 10, self.domain[2][1])
        x1t = lhs_sampling(
            mins=[self.domain[0][0] / 20, self.domain[2][0] + self.domain[2][1] / 10],
            maxs=[self.domain[0][1] / 20, self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key
        )
        local = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        data = jnp.concatenate([top, left, right, local], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_flux(self):
        key, self.key = random.split(self.key)
        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 // 2,
            key=key
        )
        data = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        return data[:, :-1], data[:, -1:]

    def sample(self, pde_name="ac"):
        return (
            self.sample_pde_rar(pde_name=pde_name),
            self.sample_ic(),
            self.sample_bc(),
            self.sample_pde(),
            self.sample_flux(),
        )


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones(2), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.adam(scheduler)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jit
def train_step(state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(pinn.loss_fn, has_aux=True, argnums=0)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)


class PFPINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        # x: (x1, x2)
        r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
        phi = (r > 0.05).astype(jnp.float32)
        c = phi.copy()
        sol = jnp.stack([phi, c], axis=1)
        return jax.lax.stop_gradient(sol)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
        phi = 1 - (1 - jnp.tanh(jnp.sqrt(cfg.OMEGA_PHI) /
                            jnp.sqrt(2 * cfg.ALPHA_PHI) * (r-0.05) * cfg.Lc)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * cfg.CSE + (1 - h_phi) * 0.0
        sol = jnp.stack([phi, c], axis=1)
        return jax.lax.stop_gradient(sol)


pinn = PFPINN(config=cfg)

init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model, model_key, cfg.LR, decay=cfg.DECAY, decay_every=cfg.DECAY_EVERY
)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
sampler = Sampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "model": pinn,
        "params": state.params,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)
stagger = StaggerSwitch(pde_names=["ac", "ch"], stagger_period=cfg.STAGGER_PERIOD)

start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    pinn.pde_name = pde_name
    pinn.causal_weightor.pde_name = pde_name

    if epoch % cfg.STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample(pde_name=pde_name)
        print(f"Epoch: {epoch}, PDE: {pde_name}")


    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        state, batch, cfg.CAUSAL_CONFIGS[pde_name + "_eps"]
    )
    if cfg.CAUSAL_WEIGHT:
        update_causal_eps(aux_vars["causal_weights"], cfg.CAUSAL_CONFIGS, pde_name)
    stagger.step_epoch()



    if epoch % cfg.STAGGER_PERIOD == 0:
        
        # save the model
        params = state.params
        model_path = f"{log_path}/model-{epoch}.npz"
        params = jax.device_get(params)
        jnp.savez(model_path, **params)
        
        fig, error = evaluate2D(
            pinn,
            state.params,
            jnp.load(f"{cfg.DATA_PATH}/mesh_points.npy"),
            cfg.DATA_PATH,
            ts=cfg.TS,
            Lc=cfg.Lc,
            Tc=cfg.Tc,
        )

        print(
            f"Epoch: {epoch}, "
            f"Error: {error:.2e}, "
            f"Loss_{pde_name}: {loss_components[0]:.2e}, "
        )

        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/{pde_name}",
                "loss/ic",
                "loss/bc",
                "loss/irr",
                "loss/flux",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                "weight/irr",
                "weight/flux",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

        if cfg.CAUSAL_WEIGHT:
            fig = pinn.causal_weightor.plot_causal_info(
                pde_name,
                aux_vars["causal_weights"],
                aux_vars["loss_chunks"],
                cfg.CAUSAL_CONFIGS[pde_name + "_eps"],
            )
            metrics_tracker.register_figure(epoch, fig, "causal_info")
            plt.close(fig)

        metrics_tracker.flush()
        
    




end_time = time.time()
print(f"Training time: {end_time - start_time}")
