"""
Phase-field simulation on a sphere — DOLFINx + cuDOLFINx (CUDA) version.

Migrated from legacy FEniCS (dolfin) to DOLFINx 0.10.x with GPU-accelerated
assembly via cuDOLFINx and GPU-resident PETSc solvers.

Physics: Allen-Cahn type equation with a surface-tension driving term on a
sphere of radius R0 inside a [-50, 50]^3 box.

Usage:
    python phase_field_cuda.py --dt 0.005 --N 64
"""

import os
import time
import argparse
import numpy as np
import math

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import fem, mesh, io

# cuDOLFINx: GPU-accelerated assembly
import cudolfinx as cufem

os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Phase-field simulation on a sphere (DOLFINx + CUDA)."
)
parser.add_argument("--dt", type=float, default=0.005, help="Initial time step size")
parser.add_argument(
    "--N", type=int, default=64, help="Number of elements in each dimension"
)
args, petsc_args = parser.parse_known_args()

# 将剩余参数传给 PETSc
PETSc.Options().insertString(" ".join(petsc_args))

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.rank

start_time = time.time()

# ---------------------------------------------------------------------------
# 物理 & 数值参数
# ---------------------------------------------------------------------------
lambda_ = 5.0
N = args.N
h = 100.0 / 64.0
epsilon = 6.0 * h / (2.0 * np.sqrt(2.0) * np.arctanh(0.9))
M = 0.1
R0 = 35.0

save_dir = f"/root/autodl-tmp/results_N{N}_dt{args.dt}"
if rank == 0 and not os.path.exists(save_dir):
    os.makedirs(save_dir)
comm.Barrier()

# ---------------------------------------------------------------------------
# 创建网格和函数空间
# ---------------------------------------------------------------------------
domain = mesh.create_box(
    comm,
    [np.array([-50.0, -50.0, -50.0]), np.array([50.0, 50.0, 50.0])],
    [N, N, N],
    cell_type=mesh.CellType.tetrahedron,
)

V = fem.functionspace(domain, ("Lagrange", 1))

if rank == 0:
    print(f"Total DOFs: {V.dofmap.index_map.size_global}")

# ---------------------------------------------------------------------------
# 函数定义
# ---------------------------------------------------------------------------
phi = fem.Function(V, name="phi")
phi_n = fem.Function(V, name="phi_n")
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)  # 用于 Jacobian

# ---------------------------------------------------------------------------
# 初始条件: tanh profile 定义球面
# ---------------------------------------------------------------------------
def initial_condition(x):
    """Vectorized initial condition for DOLFINx interpolation."""
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    return np.tanh((R0 - r) / (np.sqrt(2.0) * epsilon))


phi.interpolate(initial_condition)
phi_n.interpolate(initial_condition)

# ---------------------------------------------------------------------------
# 时间步长（用 dolfinx.fem.Constant）
# ---------------------------------------------------------------------------
dt_val = args.dt
dt = fem.Constant(domain, PETSc.ScalarType(dt_val))

# ---------------------------------------------------------------------------
# 变分形式 (残差 F, 自动求导 Jacobian J)
# ---------------------------------------------------------------------------
dx = ufl.dx

F = (
    (phi - phi_n) / dt * v * dx
    + M * ufl.dot(ufl.grad(phi), ufl.grad(v)) * dx
    + M / epsilon**2 * (phi**3 - phi) * v * dx
    + lambda_ * math.sqrt(2.0 * 0.25) * (1.0 - phi**2) / epsilon * v * dx
)

J = ufl.derivative(F, phi, u_trial)

# ---------------------------------------------------------------------------
# cuDOLFINx: 直接用 UFL form 创建 CUDA form
# ---------------------------------------------------------------------------
cuda_F_form = cufem.form(F)
cuda_J_form = cufem.form(J)
cuda_asm = cufem.CUDAAssembler()

# ---------------------------------------------------------------------------
# Newton 更新量
# ---------------------------------------------------------------------------
du = fem.Function(V)

# ---------------------------------------------------------------------------
# 配置 PETSc KSP 线性求解器 — 全 GPU 路径
#
# GMRES(200) + Jacobi，矩阵和向量都在 GPU 上。
# 足够的 restart 和 max_it 确保线性求解精度。
# ---------------------------------------------------------------------------
solver = PETSc.KSP().create(comm)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.JACOBI)
solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
solver.setGMRESRestart(200)  # 默认30太小

# 允许命令行覆盖求解器选项
solver.setFromOptions()

if rank == 0:
    print(f"[INFO] KSP type: {solver.getType()}, PC type: {solver.getPC().getType()}")

# ---------------------------------------------------------------------------
# 自定义 Newton 求解器（使用 cuDOLFINx GPU 组装）
# ---------------------------------------------------------------------------
def newton_solve(
    phi: fem.Function,
    phi_n: fem.Function,
    max_iter: int = 15,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    relaxation: float = 1.0,
):
    """
    手动实现 Newton 迭代 + 简易 line search，使用 cuDOLFINx GPU 组装。

    Returns:
        (converged: bool, iterations: int)
    """
    correction_norm_0 = None

    for i in range(max_iter):
        # --- GPU 组装残差向量 ---
        cuda_b = cuda_asm.assemble_vector(cuda_F_form)
        b_vec = cuda_b.vector  # property -> PETSc.Vec (seqcuda)
        b_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        # 检查残差范数
        res_norm = b_vec.norm(PETSc.NormType.NORM_2)

        # 残差足够小则收敛
        if i == 0:
            res_norm_0 = res_norm if res_norm > 0 else 1.0
        if res_norm < atol or res_norm / res_norm_0 < rtol:
            if rank == 0:
                print(f"  Newton iter {i}: |F|={res_norm:.6e} -> converged")
            return True, i

        b_vec.scale(-1.0)

        # --- GPU 组装 Jacobian 矩阵 ---
        cuda_A = cuda_asm.assemble_matrix(cuda_J_form)
        cuda_A.assemble()
        A_mat = cuda_A.mat  # property -> PETSc.Mat (mpiaijcusparse)

        # --- 求解 J * du = -F ---
        # du.x.petsc_vec 是 CPU 向量，b_vec 是 seqcuda (GPU)。
        # 需要创建 GPU 向量接收解，再拷贝回 du。
        solver.setOperators(A_mat)

        # 创建与 b_vec 同类型的 GPU 解向量
        x_gpu = b_vec.duplicate()
        x_gpu.set(0.0)

        solver.solve(b_vec, x_gpu)

        ksp_its = solver.getIterationNumber()
        ksp_reason = solver.getConvergedReason()
        correction_norm = x_gpu.norm(PETSc.NormType.NORM_2)

        if rank == 0:
            print(f"  Newton iter {i}: |F|={res_norm:.6e}, |du|={correction_norm:.6e}, KSP its={ksp_its}")

        # 将 GPU 解拷贝回 CPU 的 du
        # 方法: 通过 numpy array 桥接
        x_gpu_array = x_gpu.getArray()  # PETSc 会从 GPU 拷到 CPU
        du.x.array[:len(x_gpu_array)] = x_gpu_array
        du.x.scatter_forward()

        x_gpu.destroy()

        if rank == 0:
            print(f", |du|={correction_norm:.6e}, KSP its={ksp_its}, reason={ksp_reason}")

        # KSP 没收敛则 Newton 失败
        if ksp_reason < 0:
            if rank == 0:
                print(f"  KSP diverged (reason={ksp_reason}), Newton failed")
            return False, i + 1

        # --- 简易 line search: 回溯如果残差增大 ---
        alpha = relaxation
        phi_backup = phi.x.array.copy()

        for ls in range(5):
            phi.x.array[:] = phi_backup + alpha * du.x.array[:]
            phi.x.scatter_forward()

            # 计算新残差
            cuda_b_new = cuda_asm.assemble_vector(cuda_F_form)
            b_new = cuda_b_new.vector
            b_new.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            new_res = b_new.norm(PETSc.NormType.NORM_2)

            if new_res < res_norm * 1.5 or alpha < 0.05:
                break
            alpha *= 0.5
            if rank == 0:
                print(f"    line search: alpha={alpha:.4f}, |F_new|={new_res:.6e}")

        # 发散检测
        if not np.isfinite(correction_norm) or correction_norm > 1e15:
            if rank == 0:
                print(f"  Newton diverged at iter {i}")
            return False, i + 1

    return False, max_iter


# ---------------------------------------------------------------------------
# XDMF 输出（并行安全）
# ---------------------------------------------------------------------------
xdmf_file = io.XDMFFile(comm, f"{save_dir}/fields.xdmf", "w")
xdmf_file.write_mesh(domain)


def save_checkpoint(t):
    """保存当前场到 XDMF（可视化）和 .npz（评估用）。"""
    xdmf_file.write_function(phi, t)

    # 保存 numpy 数据（DOF 坐标 + 场值），评估时直接加载
    coords = V.tabulate_dof_coordinates()[:, :3]
    np.savez(
        f"{save_dir}/sol-{t:.4f}.npz",
        coords=coords,
        phi=phi.x.array.copy(),
    )


# 初始保存
now = 0.0
save_checkpoint(now)

if rank == 0:
    with open(f"{save_dir}/time_logs.txt", "w") as f:
        f.write("sim_time,compute_time\n")

# ---------------------------------------------------------------------------
# 时间推进主循环
# ---------------------------------------------------------------------------
init_dt = args.dt
total_time = 5.0
dt.value = init_dt
current_dt = init_dt

dt_max = init_dt * 2.0
dt_min = 1e-8

while now < total_time:
    # 保存当前状态以便回退
    phi.x.array[:] = phi_n.x.array[:]
    phi.x.scatter_forward()

    # Newton 求解
    converged, iterations = newton_solve(
        phi,
        phi_n,
        max_iter=15,
        atol=1e-8,
        rtol=1e-6,
        relaxation=0.9,
    )

    if not converged:
        # 求解失败 → 缩小时间步、回退、重试
        if rank == 0:
            print(
                f"Newton solver failed at t={now:.6f}. "
                f"Decreasing dt from {current_dt} to {current_dt / 2}"
            )
        current_dt /= 2.0
        dt.value = current_dt

        if current_dt < dt_min:
            if rank == 0:
                print("Time step reached minimum limit. Stopping simulation.")
            break

        # 回退 phi
        phi.x.array[:] = phi_n.x.array[:]
        phi.x.scatter_forward()
        continue

    # 成功推进
    now += current_dt

    checkpoint_time = time.time()
    if rank == 0:
        with open(f"{save_dir}/time_logs.txt", "a") as f:
            f.write(f"{now},{checkpoint_time - start_time}\n")

        print(f"Iterations: {iterations}. Step: {current_dt:.6f}. Time: {now:.6f}")

    # 更新旧解
    phi_n.x.array[:] = phi.x.array[:]
    phi_n.x.scatter_forward()

    # 每 0.1 时间单位保存一次
    if abs(now * 10 - round(now * 10)) < 1e-6:
        save_checkpoint(now)

# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
xdmf_file.close()
solver.destroy()

if rank == 0:
    end_time = time.time()
    print("Simulation finished")
    print(f"Time elapsed: {end_time - start_time:.2f} s")
    with open(f"{save_dir}/time_logs.txt", "a") as f:
        f.write(f"total_time,{end_time - start_time}\n")