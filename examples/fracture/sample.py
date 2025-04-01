import jax
import jax.numpy as jnp
from jax import random, vmap

from pinn import lhs_sampling, shifted_grid


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
            [self.n_samples * 2, self.n_samples, self.n_samples * 3],
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
            num=self.n_samples**2,
            key=key,
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
            key=key,
        )
        top = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][1], x1t[:, 1:2]],
            axis=1,
        )
        x2t = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[1][1], self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key,
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
            key=key,
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
            key=key,
        )
        data = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        return data[:, :-1], data[:, -1:]

    def sample(self, pde_name="ac"):
        return (
            self.sample_pde(),
            self.sample_ic(),
            self.sample_bc(),
            self.sample_flux(),
            self.sample_pde(),
        )
