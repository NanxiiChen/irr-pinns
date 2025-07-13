import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec

from examples.combustion.configs import Config as cfg


def evaluate1D(pinn, params, mesh, ref_path, **kwargs):
    fig, ax = plt.subplots(1,1, figsize=(8, 6))
    Lc = kwargs.get("Lc", 1.5e-3)
    mesh = mesh / Lc
    T_ref = jnp.load(f"{ref_path}/T.npy")
    T = vmap(pinn.net_T, in_axes=(None, 0))(params, mesh).squeeze()

    ax.plot(mesh, T, label="PINN", color="blue")
    ax.plot(mesh, T_ref, label="FEM", color="orange")
    ax.legend()
    error = jnp.mean((T - T_ref) ** 2) ** 0.5
    return fig, error


