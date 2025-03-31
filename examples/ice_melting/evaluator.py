import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from matplotlib.gridspec import GridSpec

from examples.ice_melting.configs import Config as cfg


def evaluate3D(pinn, params, mesh, ref_path, ts, **kwargs):
    fig = plt.figure(figsize=(3 * len(ts), 8))
    gs = GridSpec(
        4,
        len(ts) + 1,
        width_ratios=[0.1] + [1] * len(ts),
        height_ratios=[1, 1, 1, 0.3],
        figure=fig,
    )

    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim", (-0.5, 0.5))
    zlim = kwargs.get("zlim", (-0.5, 0.5))
    Lc = kwargs.get("Lc", 100)
    Tc = kwargs.get("Tc", 1.0)

    error = 0
    mesh /= Lc
    mesh = mesh[::10]

    row_names = ["PINN", "FEM", "Error"]
    for idx, row_name in enumerate(row_names):
        ax = fig.add_subplot(gs[idx, 0])
        # put row name on the left vertical axis
        ax.text(
            0.5,
            0.5,
            row_name,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
        )
        ax.set_axis_off()

    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic / Tc
        pred = vmap(pinn.net_u, in_axes=(None, 0, 0))(params, mesh, t).squeeze()

        ax = fig.add_subplot(gs[0, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = (pred > -0.5) & (pred < 0.5)
        ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=pred[interface_idx],
            cmap="coolwarm",
            label="phi",
            vmin=-1,
            vmax=1,
        )
        r_pinn = (
            jnp.sqrt(
                mesh[interface_idx, 0] ** 2
                + mesh[interface_idx, 1] ** 2
                + mesh[interface_idx, 2] ** 2
            )
            * Lc
        )

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title=f"t={tic}",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        ax.set_axis_off()
        ax.invert_zaxis()

        # the numerical solution by FEM
        # ref_sol = jnp.load(f"{ref_path}/sol-{tic:.4f}.npy")[::10]

        # or we can calculate the sol using the analytical solution
        Rt = cfg.R0 - cfg.LAMBDA * tic
        Rxyz = jnp.sqrt(mesh[:, 0] ** 2 + mesh[:, 1] ** 2 + mesh[:, 2] ** 2) * Lc
        ref_sol = jnp.tanh((Rt - Rxyz) / (jnp.sqrt(2) * cfg.EPSILON))
        diff = jnp.abs(pred - ref_sol)

        ax = fig.add_subplot(gs[1, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = (ref_sol > -0.5) & (ref_sol < 0.5)
        ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=ref_sol[interface_idx],
            cmap="coolwarm",
            label="phi",
            vmin=-1,
            vmax=1,
        )
        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, idx + 1], projection="3d", box_aspect=(1, 1, 1))
        interface_idx = diff > 0.05
        error_bar = ax.scatter(
            mesh[interface_idx, 0],
            mesh[interface_idx, 1],
            mesh[interface_idx, 2],
            c=jnp.abs(pred[interface_idx] - ref_sol[interface_idx]),
            cmap="coolwarm",
            label="error",
        )

        ax.set(
            xlabel="x",
            ylabel="y",
            zlabel="z",
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
        error += jnp.mean(diff**2)

        ax.set_axis_off()
        ax.invert_zaxis()

        interface_idx = jnp.where((ref_sol > -0.5) & (ref_sol < 0.5))[0]
        r_fem = (
            jnp.sqrt(
                mesh[interface_idx, 0] ** 2
                + mesh[interface_idx, 1] ** 2
                + mesh[interface_idx, 2] ** 2
            )
            * Lc
        )
        r_analytical = cfg.R0 - cfg.LAMBDA * tic

        ax.text2D(
            0.05,
            -0.2,
            f"R_analytical = {r_analytical:.2f}\n"
            f"R_pinn = {jnp.mean(r_pinn):.2f}\n"
            f"R_fem = {jnp.mean(r_fem):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

        ax = fig.add_subplot(gs[3, idx + 1])
        fig.colorbar(error_bar, ax=ax, orientation="horizontal")
        ax.set_axis_off()

    error /= len(ts)
    return fig, error

