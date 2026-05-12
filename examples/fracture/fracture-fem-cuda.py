"""
Phase-field fracture simulation — DOLFINx + PETSc CUDA version.

Migrated from legacy FEniCS (dolfin). Staggered scheme:
  1. Solve displacement u (linear, vector field)
  2. Solve phase field phi (linear, scalar field)
  3. Iterate until stagger convergence

For linear problems with Dirichlet BCs, standard DOLFINx CPU assembly
is used (cuDOLFINx 不支持在 GPU 矩阵上直接施加边界条件).
GPU acceleration via PETSc CUDA-aware KSP solvers (GAMG, etc).

Usage:
    python fracture_cuda.py
    python fracture_cuda.py -ksp_type gmres -pc_type gamg
"""

import os
import time
import argparse
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)

os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# 参数解析（允许 PETSc 选项透传）
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Phase-field fracture (DOLFINx + CUDA).")
parser.add_argument("--mesh", type=str, default="mesh.xdmf", help="Mesh file")
args, petsc_args = parser.parse_known_args()
PETSc.Options().insertString(" ".join(petsc_args))

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank

start_time = time.time()

# ---------------------------------------------------------------------------
# 读取网格
# ---------------------------------------------------------------------------
with io.XDMFFile(comm, args.mesh, "r") as infile:
    domain = infile.read_mesh(name="Grid")

tdim = domain.topology.dim
fdim = tdim - 1

if rank == 0:
    print(f"Mesh: {domain.topology.index_map(tdim).size_global} cells, dim={tdim}")

# ---------------------------------------------------------------------------
# 材料参数
# ---------------------------------------------------------------------------
Gc = 2.7
l = 0.024
lmbda = 121.1538e3
mu = 80.7692e3

# ---------------------------------------------------------------------------
# 函数空间
# ---------------------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))              # phi (标量)
W = fem.functionspace(domain, ("Lagrange", 1, (tdim,)))      # u (向量)
WW = fem.functionspace(domain, ("DG", 0))                    # H (history)

if rank == 0:
    print(f"Scalar DOFs (phi): {V.dofmap.index_map.size_global}")
    print(f"Vector DOFs (u):   {W.dofmap.index_map.size_global}")

# ---------------------------------------------------------------------------
# 本构关系
# ---------------------------------------------------------------------------
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(tdim)

def psi_plus(u):
    """Tension part of elastic energy."""
    return 0.5 * (lmbda + mu) * (
        0.5 * (ufl.tr(epsilon(u)) + abs(ufl.tr(epsilon(u))))
    ) ** 2 + mu * ufl.inner(ufl.dev(epsilon(u)), ufl.dev(epsilon(u)))

# ---------------------------------------------------------------------------
# 边界定位
# ---------------------------------------------------------------------------
def top_boundary(x):
    return np.isclose(x[1], 0.5)

def bot_boundary(x):
    return np.isclose(x[1], -0.5)

def crack_boundary(x):
    return (np.abs(x[1]) < 1e-3) & (x[0] <= 0.0)

# ---------------------------------------------------------------------------
# 位移边界条件
# ---------------------------------------------------------------------------
# bottom: u = (0, 0)
bot_dofs = fem.locate_dofs_geometrical(W, bot_boundary)
bc_bot = fem.dirichletbc(
    np.array([0.0, 0.0], dtype=default_scalar_type), bot_dofs, W
)

# top: ux = 0
W0, _ = W.sub(0).collapse()
top_dofs_x = fem.locate_dofs_geometrical((W.sub(0), W0), top_boundary)
zero_func = fem.Function(W0)
zero_func.x.array[:] = 0.0
bc_top_ux = fem.dirichletbc(zero_func, top_dofs_x, W.sub(0))

# top: uy = load (time-dependent)
W1, _ = W.sub(1).collapse()
load_func = fem.Function(W1)
top_dofs_y = fem.locate_dofs_geometrical((W.sub(1), W1), top_boundary)
bc_top_uy = fem.dirichletbc(load_func, top_dofs_y, W.sub(1))

bc_u = [bc_bot, bc_top_ux, bc_top_uy]

# 相场边界条件: phi = 1 on initial crack
crack_dofs = fem.locate_dofs_geometrical(V, crack_boundary)
bc_phi = [fem.dirichletbc(PETSc.ScalarType(1.0), crack_dofs, V)]

# ---------------------------------------------------------------------------
# 函数
# ---------------------------------------------------------------------------
unew = fem.Function(W, name="u")
uold = fem.Function(W, name="u_old")
pnew = fem.Function(V, name="phi")
pold = fem.Function(V, name="phi_old")
Hold = fem.Function(WW, name="H")

# ---------------------------------------------------------------------------
# 初始条件: phi = exp(-|y|/l) for x <= 0
# ---------------------------------------------------------------------------
def initial_crack(x):
    vals = np.zeros(x.shape[1])
    mask = x[0] <= 0.0
    vals[mask] = np.exp(-np.abs(x[1][mask]) / l)
    return vals

pnew.interpolate(initial_crack)
pold.interpolate(initial_crack)

# ---------------------------------------------------------------------------
# 变分形式
# ---------------------------------------------------------------------------
u_trial = ufl.TrialFunction(W)
v_test = ufl.TestFunction(W)
p_trial = ufl.TrialFunction(V)
q_test = ufl.TestFunction(V)

dx = ufl.dx

# 位移方程: (1-phi_old)^2 * sigma(u) : grad(v) = 0
a_u = ((1.0 - pold) ** 2) * ufl.inner(ufl.grad(v_test), sigma(u_trial)) * dx
L_u = ufl.dot(fem.Constant(domain, PETSc.ScalarType((0.0, 0.0))), v_test) * dx

# 相场方程: Gc*l * grad(p).grad(q) + (Gc/l + 2*H) * p*q = 2*H * q
a_phi = (
    Gc * l * ufl.inner(ufl.grad(p_trial), ufl.grad(q_test))
    + (Gc / l + 2.0 * Hold) * ufl.inner(p_trial, q_test)
) * dx
L_phi = 2.0 * Hold * q_test * dx

# 编译
a_u_form = fem.form(a_u)
L_u_form = fem.form(L_u)
a_phi_form = fem.form(a_phi)
L_phi_form = fem.form(L_phi)

# ---------------------------------------------------------------------------
# KSP 求解器
# ---------------------------------------------------------------------------
def create_ksp():
    ksp = PETSc.KSP().create(comm)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.GAMG)
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    ksp.setGMRESRestart(200)
    ksp.setFromOptions()
    return ksp

solver_u = create_ksp()
solver_phi = create_ksp()

if rank == 0:
    print(f"[INFO] KSP: {solver_u.getType()}, PC: {solver_u.getPC().getType()}")

# ---------------------------------------------------------------------------
# History field 更新
# ---------------------------------------------------------------------------
def update_history_field():
    """H = max(psi_plus(unew), Hold)，投影到 DG0。"""
    psi_expr = fem.Expression(
        psi_plus(unew),
        WW.element.interpolation_points(),
    )
    psi_val = fem.Function(WW)
    psi_val.interpolate(psi_expr)
    Hold.x.array[:] = np.maximum(psi_val.x.array[:], Hold.x.array[:])
    Hold.x.scatter_forward()

# ---------------------------------------------------------------------------
# 线性求解
# ---------------------------------------------------------------------------
def solve_linear(a_form, L_form, bcs, solution, ksp):
    """标准 DOLFINx 组装 + BC + KSP 求解。"""
    A = create_matrix(a_form)
    A.zeroEntries()
    assemble_matrix(A, a_form, bcs=bcs)
    A.assemble()

    b = create_vector(L_form)
    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L_form)
    apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    set_bc(b, bcs)

    ksp.setOperators(A)
    ksp.solve(b, solution.x.petsc_vec)
    solution.x.scatter_forward()

    its = ksp.getIterationNumber()
    A.destroy()
    b.destroy()
    return its

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
save_dir = "/root/autodl-tmp/fracture/ResultsNumpy-cuda"
os.makedirs(save_dir, exist_ok=True)

xdmf_phi = io.XDMFFile(comm, f"{save_dir}/phi.xdmf", "w")
xdmf_phi.write_mesh(domain)

# Traction 计算: 标记 top 边界面
domain.topology.create_connectivity(fdim, tdim)
top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
mt = mesh.meshtags(
    domain, fdim, top_facets,
    np.full(len(top_facets), 1, dtype=np.int32),
)
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
n = ufl.FacetNormal(domain)

# 保存网格坐标
coords_V = V.tabulate_dof_coordinates()[:, :tdim]
np.save(f"{save_dir}/mesh_points.npy", coords_V)

# 初始保存
np.savez(
    f"{save_dir}/sol-{0.0:.4f}.npz",
    phi=pold.x.array.copy(),
    u=uold.x.array.copy(),
)

# ---------------------------------------------------------------------------
# 时间推进
# ---------------------------------------------------------------------------
t = 0.0
u_r = 0.007
deltaT = 0.05
tol = 1e-3

if rank == 0:
    with open(f"{save_dir}/ForcevsDisp.txt", "w") as f:
        f.write("Force\tDisp\n")
    with open(f"{save_dir}/TimeLog.csv", "w") as f:
        f.write("Time,ComputationTime\n")

# ---------------------------------------------------------------------------
# Staggered scheme 主循环
# ---------------------------------------------------------------------------
while t <= 0.78:
    t += deltaT
    if t >= 0.70:
        deltaT = 0.0001

    if rank == 0:
        check_time = time.time()
        with open(f"{save_dir}/TimeLog.csv", "a") as f:
            f.write(f"{t:.4f},{check_time - start_time:.4f}\n")
        print(f"Time step: {t:.4f}")

    # 施加载荷
    load_func.x.array[:] = t * u_r
    load_func.x.scatter_forward()

    iter_count = 0
    err = 1.0

    while err > tol:
        iter_count += 1

        # 1. 求解位移
        its_u = solve_linear(a_u_form, L_u_form, bc_u, unew, solver_u)

        # 2. 求解相场
        its_phi = solve_linear(a_phi_form, L_phi_form, bc_phi, pnew, solver_phi)

        # 3. 误差 (L2 norm of increment)
        err_u_local = fem.assemble_scalar(
            fem.form(ufl.inner(unew - uold, unew - uold) * dx)
        )
        err_u = np.sqrt(abs(comm.allreduce(err_u_local, op=MPI.SUM)))

        err_phi_local = fem.assemble_scalar(
            fem.form(ufl.inner(pnew - pold, pnew - pold) * dx)
        )
        err_phi = np.sqrt(abs(comm.allreduce(err_phi_local, op=MPI.SUM)))

        err = max(err_u, err_phi)

        # 4. 更新旧解
        uold.x.array[:] = unew.x.array[:]
        uold.x.scatter_forward()
        pold.x.array[:] = pnew.x.array[:]
        pold.x.scatter_forward()

        # 5. 更新 history field
        update_history_field()

        if err < tol:
            if rank == 0:
                print(
                    f"  Converged: iter={iter_count}, "
                    f"err_u={err_u:.4e}, err_phi={err_phi:.4e}, "
                    f"KSP(u)={its_u}, KSP(phi)={its_phi}"
                )

            if round(t * 1e4) % 10 == 0:
                xdmf_phi.write_function(pnew, t)

                np.savez(
                    f"{save_dir}/sol-{t:.4f}.npz",
                    phi=pnew.x.array.copy(),
                    u=unew.x.array.copy(),
                )

                # Traction force on top boundary
                Traction = ufl.dot(sigma(unew), n)
                fy_form = fem.form(Traction[1] * ds(1))
                fy_local = fem.assemble_scalar(fy_form)
                fy = comm.allreduce(fy_local, op=MPI.SUM)

                if rank == 0:
                    with open(f"{save_dir}/ForcevsDisp.txt", "a") as f:
                        f.write(f"{t * u_r}\t{fy}\n")

# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
xdmf_phi.close()
solver_u.destroy()
solver_phi.destroy()

if rank == 0:
    end_time = time.time()
    print("Simulation completed")
    print(f"Total computation time: {end_time - start_time:.2f} s")
    with open(f"{save_dir}/TimeLog.csv", "a") as f:
        f.write(f"Total,{end_time - start_time:.4f}\n")