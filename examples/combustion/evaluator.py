import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap

from examples.combustion.configs import Config as cfg



def evaluate1D(pinn, params, mesh, ref_path, **kwargs):
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    Lc = kwargs.get("Lc", 1.5e-3)
    mesh = mesh / Lc

    ax = axes[0]
    T_ref = jnp.load(f"{ref_path}/T.npy")
    T = vmap(pinn.net_T, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, T, label="PINN")
    ax.plot(mesh, T_ref, label="FEM", ls="--")
    ax.legend()
    # error = jnp.mean((T - T_ref) ** 2) ** 0.5
    l2_error = jnp.linalg.norm(T - T_ref) / jnp.linalg.norm(T_ref)

    ax = axes[1]
    yf_ref = jnp.load(f"{ref_path}/YF.npy")
    yf = vmap(pinn.net_yf, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, yf, label="PINN")
    ax.plot(mesh, yf_ref, label="FEM", ls="--")
    ax.legend()

    ax = axes[2]
    u_ref = jnp.load(f"{ref_path}/u.npy")
    u = vmap(pinn.net_u, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, u, label="PINN")
    ax.plot(mesh, u_ref, label="FEM", ls="--")
    ax.legend()

    ax = axes[3]
    rho_ref = jnp.load(f"{ref_path}/rho.npy")
    rho = vmap(pinn.net_rho, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, rho, label="PINN")
    ax.plot(mesh, rho_ref, label="FEM", ls="--")
    ax.legend()

    ax = axes[4]
    omega_ref = jnp.load(f"{ref_path}/omega.npy")
    omega = vmap(pinn.net_omega, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, omega, label="PINN")
    ax.plot(mesh, omega_ref, label="FEM", ls="--")
    ax.legend()

    ax = axes[5]
    p_ref = jnp.load(f"{ref_path}/p.npy")
    p = vmap(pinn.net_p, in_axes=(None, 0))(params, mesh).squeeze()
    ax.plot(mesh, p, label="PINN")
    ax.plot(mesh, p_ref, label="FEM", ls="--")
    ax.legend()
    
    return fig, l2_error


