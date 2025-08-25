from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from pinn import CausalWeightor, MLP, ModifiedMLP, ResNet


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
        **kwargs,
    ):
        super().__init__()

        loss_terms = kwargs.get("loss_terms", [])
        self.loss_fn_panel = [getattr(self, f"loss_{term}") for term in loss_terms]
        self.cfg = config
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP, "resnet": ResNet}
        self.model = arch[self.cfg.ARCH_NAME](
            act_name=self.cfg.ACT_NAME,
            num_layers=self.cfg.NUM_LAYERS,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUT_DIM,
            fourier_emb=self.cfg.FOURIER_EMB,
            emb_scale=self.cfg.EMB_SCALE,
            emb_dim=self.cfg.EMB_DIM,
        )
        self.causal_weightor = causal_weightor


    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        u = self.model.apply(params, x, t)
        return u


    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        # $$\frac{\partial^2 p}{\partial t^2} + 2\alpha\frac{\partial p}{\partial t} = c^2\frac{\partial^2 p}{\partial x^2}$$
        Tc = self.cfg.Tc
        Lc = self.cfg.Lc
        du_dt = jax.jacrev(self.net_u, argnums=2)(params, x, t)[0] / Tc
        d2u_dt2 = jax.hessian(self.net_u, argnums=2)(params, x, t)[0][0] / (Tc**2)
        d2u_dx2 = jax.hessian(self.net_u, argnums=1)(params, x, t)[0][0] / (Lc**2)
        alpha = self.cfg.ALPHA
        c = self.cfg.VELOCITY
        pde = d2u_dt2 + alpha * du_dt - c**2 * d2u_dx2
        return pde.squeeze()


    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch, eps, *args, **kwargs):
        x, t = batch
        residual = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(residual**2), {}
        else:
            causal_data = jnp.stack((t.reshape(-1), ), axis=0)
            return self.causal_weightor.compute_causal_loss(residual, causal_data, eps)

    @partial(jit, static_argnums=(0,))
    def net_irr(self, params, x, t, eps=1e-6):
        
        @jax.jit
        def rms_position_over_time(params, x, t):
            # x: a fix position
            # t: time sequence
            u = vmap(self.net_u, in_axes=(None, None, 0))(params, x, t)
            return jnp.mean(u**2)
        
        drms_dx = jax.grad(rms_position_over_time, argnums=1)(params, x, t)
        # drms_dx should be non-positive, we penalize its positive part
        irr = jax.nn.softplus(drms_dx)

        return irr

    # def loss_ic(self, params, batch, *args, **kwargs):
    #     x, t = batch
    #     u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
    #     ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
    #     return jnp.mean((u - ref) ** 2), {}

    # def loss_bc(self, params, batch, *args, **kwargs):
    #     x, t = batch
    #     u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
    #     ref = vmap(self.ref_sol_bc, in_axes=(0, 0))(x, t)
    #     return jnp.mean((u - ref) ** 2), {}

    def loss_irr(self, params, batch, *args, **kwargs):
        x, t = batch
        irr = vmap(self.net_irr, in_axes=(None, 0, None))(params, x, t)
        # irr should be non-positive
        return jnp.mean(irr**2), {}

    @partial(jit, static_argnums=(0,))
    def compute_losses_and_grads(self, params, batch, eps):
        if len(batch) != len(self.loss_fn_panel):
            raise ValueError(
                "The number of loss functions "
                "should be equal to the number of items in the batch"
            )
        losses = []
        grads = []
        aux_vars = {}
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):

            (loss_item, aux), grad_item = jax.value_and_grad(
                loss_item_fn, has_aux=True
            )(params, batch_item, eps)
            aux_vars.update(aux)
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads, aux_vars

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch, eps,):
        losses, grads, aux_vars = self.compute_losses_and_grads(
            params, batch, eps,
        )

        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)
        # weights = weights.at[1].set(weights[1] * 5)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-8):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)

        return jax.lax.stop_gradient(weights)


    