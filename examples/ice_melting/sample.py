import jax.numpy as jnp
from jax import random

from pinn import lhs_sampling, shifted_grid, Sampler


class IceMeltingSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((-0.4, 0.4), (-0.4, 0.4), (0, 0.4), (0, 1)),
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
            [self.n_samples, self.n_samples, self.n_samples, self.n_samples * 3],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[2][1]],
            num=10000,
            key=key,
        )
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample(self, *args, **kwargs):
        data = (
            self.sample_pde_rar(*args, **kwargs),
            self.sample_ic(),
            self.sample_pde(),
        )
        return data
