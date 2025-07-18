import jax
import jax.numpy as jnp
from jax import random, vmap

from pinn import lhs_sampling, shifted_grid, Sampler


class CombustionSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((0, 1.0),),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [self.n_samples,],
            key,
        )
        return data
    
    def sample_pde_rar(self, fns, params, *args, **kwargs):
        key, self.key = random.split(self.key)
        grid_key, lhs_key = random.split(key)
        common_points = self.sample_pde()
        adaptive_base = lhs_sampling(
            self.mins,
            self.maxs,
            self.adaptive_kw["num"] * self.adaptive_kw["ratio"],
            key,
        )
        rar_points = jnp.zeros(
            (self.adaptive_kw["num"] * len(fns), adaptive_base.shape[1]),
        )
        for idx, fn in enumerate(fns):
            res = jax.lax.stop_gradient(
                vmap(fn, in_axes=(None, 0))(
                    params,
                    adaptive_base,
                ),
            )
            res = res.squeeze(axis=-1)
            _, indices = jax.lax.top_k(jnp.abs(res), self.adaptive_kw["num"])
            selected_points = adaptive_base[indices]
            rar_points = rar_points.at[
                idx * self.adaptive_kw["num"] : (idx + 1) * self.adaptive_kw["num"]
            ].set(selected_points)

        data = jnp.concatenate([
            common_points,
            rar_points,
        ], axis=0)
        return data
        

    
    def sample_bc(self):
        left, right = self.domain[0]
        left = jnp.array([[left,],])
        right = jnp.array([[right,],])
        return left, right
    
    def sample(self, *args, **kwargs):
        left, right = self.sample_bc()
        if kwargs.get("rar", False):
            pde = self.sample_pde_rar(*args, **kwargs)
        else:
            pde = self.sample_pde()
        return (
            pde, # for PDE loss
            left, left,
            pde # for irreversible loss
        )