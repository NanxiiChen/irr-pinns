import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec


def evaluate1D(pinn, params, data_path, ts, xlim):
    mesh = jnp.load(f"{data_path}/fisher_mesh.npy")
    times = jnp.load(f"{data_path}/fisher_times.npy")
    sol = jnp.load(f"{data_path}/fisher_sol.npy")
    xx, tt = jnp.meshgrid(mesh, times)


    fig, axes = plt.subplots(1, 2, figsize=(10, 5), 
                             subplot_kw={"projection": "3d"})
    ax = axes[0]
    ax.plot_surface(xx, tt, sol, cmap="viridis")

    # prediction
    xx_flat = xx.flatten()[:, None]
    tt_flat = tt.flatten()[:, None]
    pred_flat = vmap(
        pinn.net_u, in_axes=(None, 0, 0)
    )(params, xx_flat, tt_flat)

    pred = pred_flat.reshape(xx.shape)
    ax = axes[1]
    ax.plot_surface(xx, tt, pred, cmap="viridis")

    error = jnp.mean((sol - pred) ** 2)
    return fig, error

