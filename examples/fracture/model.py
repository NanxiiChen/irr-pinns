from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, random, vmap

from pinn import CausalWeightor, MLP, ModifiedMLP


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
        causal_weightor: CausalWeightor = None,
    ):
        super().__init__()

        self.cfg = config
        self.flux_idx = 1
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_irr,
        ]
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
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

        self.loss_fn_ac = partial(self.loss_fn, pde_name="ac")
        self.loss_fn_ch = partial(self.loss_fn, pde_name="ch")

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        # x: (d,), t: (1,)
        # phi: scalar, disp: (d,)
        phi, *disp = self.model.apply(params, x, t)
        phi = nn.tanh(phi) / 2 + 0.5
        disp = disp / self.cfg.DISP_PRE_SCALE
        return phi, disp

    def epsilon(self, params, x, t):
        # epsilon: (d, d)
        # $$ \varepsilon = sym(\nabla u) = \frac{1}{2}(\nabla u + (\nabla u)^T) $$
        disp_fn = lambda x, t: self.net_u(params, x, t)[1]
        nabla_disp = jax.jacrev(disp_fn, argnums=0)(x, t)
        assert nabla_disp.shape == (x.shape[-1], x.shape[-1])
        return (nabla_disp + nabla_disp.T) / 2

    def sigma(self, params, x, t):
        # sigma: (d, d)
        return 2.0 * self.cfg.MU * self.epsilon(
            params, x, t
        ) + self.cfg.LMBDA * jnp.trace(self.epsilon(params, x, t)) * jnp.eye(
            x.shape[-1]
        )

    def psi(self, params, x, t):
        # psi: scalar
        epsilon = self.epsilon(params, x, t)
        return self.cfg.LMBDA * jnp.trace(epsilon) ** 2 / 2 + self.cfg.MU * jnp.sum(
            epsilon**2
        )

    @partial(jit, static_argnums=(0,))
    def net_balance(self, params, x, t):
        # $$ (1 - \phi) **2 \nabla \cdot \sigma = 0 $$
        phi, disp = self.net_u(params, x, t)
        # 求应力张量的散度
        jac_sigma_x = jax.jacrev(self.sigma, argnums=0)(params, x, t)
        div_sigma = jnp.sum(jac_sigma_x, axis=-1)
        
        balance = (1 - phi) ** 2 * div_sigma
        
        return balance
    
    
    @partial(jit, static_argnums=(0,))
    def net_pf(self, params, x, t):
        phi, disp = self.net_u(params, x, t)
        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )
        lap_phi = jnp.linalg.trace(hess_phi_x)
        
        pf = (
            self.cfg.GC * (phi / self.cfg.L - self.cfg.L * lap_phi)
            - 2 * (1 - phi) * self.psi(params, x, t)
        ) 
        return pf
        
        
        
        
        

    @partial(jit, static_argnums=(0,))
    def net_ac(self, params, x, t):
        AC1 = 2 * self.cfg.AA * self.cfg.LP * self.cfg.Tc
        AC2 = self.cfg.LP * self.cfg.OMEGA_PHI * self.cfg.Tc
        AC3 = self.cfg.LP * self.cfg.ALPHA_PHI * self.cfg.Tc / self.cfg.Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)
        h_phi = -2 * phi**3 + 3 * phi**2
        dh_dphi = -6 * phi**2 + 6 * phi
        dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

        jac_phi_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=1)
        dphi_dt = jac_phi_t(x, t)[0]

        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        lap_phi = jnp.linalg.trace(hess_phi_x(x, t))

        ac = (
            dphi_dt
            - AC1
            * (c - h_phi * (self.cfg.CSE - self.cfg.CLE) - self.cfg.CLE)
            * (self.cfg.CSE - self.cfg.CLE)
            * dh_dphi
            + AC2 * dg_dphi
            - AC3 * lap_phi
        )
        return ac / self.cfg.AC_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_ch(self, params, x, t):
        CH1 = 2 * self.cfg.AA * self.cfg.MM * self.cfg.Tc / self.cfg.Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)

        jac_phi_x = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        nabla_phi = jac_phi_x(x, t)

        jac_c_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=1)
        dc_dt = jac_c_t(x, t)[0]

        # hess_phi_x, hess_c_x = jax.hessian(self.net_u, argnums=(1))(params, x, t)

        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )
        hess_c_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )
        # hess_phi_x, hess_c_x = jax.hessian(self.net_u, argnums=(1))(params, x, t)

        lap_phi = jnp.linalg.trace(hess_phi_x)
        lap_c = jnp.linalg.trace(hess_c_x)

        lap_h_phi = 6 * (
            phi * (1 - phi) * lap_phi + (1 - 2 * phi) * jnp.sum(nabla_phi**2)
        )

        ch = dc_dt - CH1 * lap_c + CH1 * (self.cfg.CSE - self.cfg.CLE) * lap_h_phi

        return ch / self.cfg.CH_PRE_SCALE

    def net_speed(self, params, x, t):
        jac_dt = jax.jacrev(self.net_u, argnums=2)
        dphi_dt, dc_dt = jac_dt(params, x, t)
        return dphi_dt, dc_dt

    def ref_sol_ic(self, x, t):
        raise NotImplementedError

    def ref_sol_bc(self, x, t):
        raise NotImplementedError

    def net_nabla(
        self,
        params,
        x,
        t,
    ):
        idx = self.flux_idx
        nabla_phi_part = jax.jacrev(
            lambda x, t: self.net_u(params, x, t)[0], argnums=0
        )(x, t)[idx]
        nabla_c_part = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )[idx]
        return nabla_phi_part, nabla_c_part

    # def loss_ac(self, params, batch, eps):
    #     x, t = batch
    #     ac = vmap(self.net_ac, in_axes=(None, 0, 0))(params, x, t)
    #     if not self.cfg.CAUSAL_WEIGHT:
    #         return jnp.mean(ac**2), {}
    #     else:
    #         return self.causal_weightor.compute_causal_loss(ac, t, eps)

    # def loss_ch(self, params, batch, eps):
    #     x, t = batch
    #     ch = vmap(self.net_ch, in_axes=(None, 0, 0))(params, x, t)
    #     if not self.cfg.CAUSAL_WEIGHT:
    #         return jnp.mean(ch**2), {}
    #     else:
    #         return self.causal_weightor.compute_causal_loss(ch, t, eps)

    @partial(jit, static_argnums=(0, 4))
    def loss_pde(self, params, batch, eps, pde_name: str):
        x, t = batch
        residual = jax.lax.cond(
            pde_name == "ac",
            lambda operand: vmap(self.net_ac, in_axes=(None, 0, 0))(params, *operand),
            lambda operand: vmap(self.net_ch, in_axes=(None, 0, 0))(params, *operand),
            operand=(x, t),
        )
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(residual**2), {}
        else:
            return self.causal_weightor.compute_causal_loss(residual, t, eps)

    def loss_ic(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    def loss_bc(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt, dc_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(dphi_dt)) + jnp.mean(jax.nn.relu(dc_dt))

    def loss_flux(self, params, batch):
        x, t = batch
        dphi_dx, dc_dx = vmap(self.net_nabla, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(dphi_dx**2) + jnp.mean(dc_dx**2)

    @partial(jit, static_argnums=(0, 4))
    def compute_losses_and_grads(self, params, batch, eps, pde_name):
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
            if idx == 0:
                (loss_item, aux), grad_item = jax.value_and_grad(
                    loss_item_fn, has_aux=True
                )(params, batch_item, eps, pde_name)
                aux_vars.update(aux)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads, aux_vars

    @partial(jit, static_argnums=(0, 4))
    def loss_fn(self, params, batch, eps, pde_name):
        losses, grads, aux_vars = self.compute_losses_and_grads(
            params, batch, eps, pde_name
        )

        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
