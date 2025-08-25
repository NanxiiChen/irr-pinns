import jax.numpy as jnp
from jax import random

from pinn import lhs_sampling, shifted_grid, Sampler


class WaveSampler(Sampler):
    def __init__(
        self,
        n_samples,
        domain=((-1.0, 1.0), (0, 1)),
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
            [self.n_samples, self.n_samples],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0],],
            maxs=[self.domain[0][1],],
            num=self.n_samples*20,
            key=key,
        )
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)
        ts = lhs_sampling(
            mins=[self.domain[1][0],],
            maxs=[self.domain[1][1],],
            num=self.n_samples*100,
            key=key,
        )
        x_left = jnp.ones((self.n_samples*100, 1)) * self.domain[0][0]
        x_right = jnp.ones((self.n_samples*100, 1)) * self.domain[0][1]
        return {
            "left": (x_left, ts),
            "right": (x_right, ts),
        }

    def sample_irr(self):
        # do not need to do cross product here
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0],],
            maxs=[self.domain[0][1],],
            num=self.n_samples,
            key=key,
        )
        t = lhs_sampling(
            mins=[self.domain[1][0],],
            maxs=[self.domain[1][1],],
            num=self.n_samples,
            key=self.key,
        )
        return x, t
    

    def sample(self, *args, **kwargs):
        pde = self.sample_pde()
        ic = self.sample_ic()
        bc = self.sample_bc()
        irr = self.sample_irr()
        return (
            pde,
            ic,
            bc["left"], 
            bc["left"],
            irr,
        )
