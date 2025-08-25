import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec


def evaluate1D(pinn, params, data_path, xlim, *args, **kwargs):
    mesh = jnp.load(f"{data_path}/wave_mesh.npy").reshape(-1, 1)
    times = jnp.load(f"{data_path}/wave_times.npy").reshape(-1, 1)
    sol = jnp.load(f"{data_path}/wave_sol.npy")

    Tc = kwargs.get("Tc", 10.0)
    Lc = kwargs.get("Lc", 1.0)
    times_inp = times / Tc
    mesh_inp = mesh / Lc
    


    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    ax = axes[0]
    # plot the left boundary over time
    x_inp = jnp.full_like(times_inp, xlim[0])
    pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, x_inp, times_inp)
    ax.plot(times, pred, label="PINN",)
    ax.plot(times, sol[:, 0], label="FEM", linestyle="dashed")
    ax.set_title("Wave at Left Boundary Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wave Amplitude")
    ax.legend()

    ax = axes[1]
    # plot the right boundary over time
    x_inp = jnp.full_like(times_inp, xlim[1])
    pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, x_inp, times_inp)
    ax.plot(times, pred, label="PINN",)
    ax.plot(times, sol[:, -1], label="FEM", linestyle="dashed")
    ax.set_title("Wave at Right Boundary Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wave Amplitude")
    ax.legend()

    ax = axes[2]
    # plot the wave at the end of simulation over space
    t_inp = jnp.full_like(mesh_inp, times_inp[-1])
    pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, mesh_inp, t_inp)
    ax.plot(mesh, pred, label="PINN",)
    ax.plot(mesh, sol[-1, :], label="FEM", linestyle="dashed")
    ax.set_title("Wave at End of Simulation Over Space")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Wave Amplitude")
    ax.legend()

    ax = axes[3]
    # plot the RMS of each spatial point over time
    def rms_func(x, t):
        pred = vmap(pinn.net_u, in_axes=(None, None, 0))(params, x, t)
        return jnp.mean(pred**2)
    rms_values = vmap(rms_func, in_axes=(0, None))(mesh_inp, times_inp)
    rms_refs = jnp.mean(sol**2, axis=0)
    ax.plot(mesh, rms_values, label="PINN",)
    ax.plot(mesh, rms_refs, label="FEM", linestyle="dashed")
    ax.set_title("RMS of Wave Over Space")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("RMS Amplitude")
    ax.legend()

    ax = axes[4]
    # plot the image of the wave over space and time
    ax.imshow(
        sol,
        extent=[mesh.min(), mesh.max(), times.max(), times.min()],
        aspect="auto", cmap="jet",
        vmin=-1, vmax=1
    )
    ax.set_title("FEM Wave Over Space and Time")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Time")

    ax = axes[5]
    xx, tt = jnp.meshgrid(mesh_inp.flatten(), times_inp.flatten())
    x_inp = xx.flatten().reshape(-1, 1)
    t_inp = tt.flatten().reshape(-1, 1)
    pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, x_inp, t_inp)
    pred = pred.reshape(xx.shape)
    ax.imshow(
        pred,
        extent=[mesh.min(), mesh.max(), times.max(), times.min()],
        aspect="auto", cmap="jet",
        vmin=-1, vmax=1
    )
    ax.set_title("PINN Wave Over Space and Time")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Time")

    error = jnp.mean((jnp.abs(pred - sol))**2)


    return fig, error





