"""
1D Fisher equation solver — DOLFINx + cuDOLFINx (CUDA) version.

Migrated from legacy FEniCS (dolfin) to DOLFINx 0.10.x with GPU-accelerated
assembly via cuDOLFINx and GPU-resident PETSc solvers.

PDE:  ∂u/∂t = ν Δu - σ u - φ u² - ψ u³
      (σ = -r,  φ = r·α,  ψ = -1,  ν = 1)

Initial condition: u(x,0) = exp(-x²)
Boundary condition: u = 0 on ∂Ω (Dirichlet)
Domain: [x_left, x_right] = [-3, 3]

Usage:
    python fisher_equation_cuda.py --r 1.0 --alpha 1.0
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
from dolfinx import fem, mesh, io

# cuDOLFINx: GPU-accelerated assembly
import cudolfinx as cufem

os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="求解一维Fisher方程 (DOLFINx + CUDA)")
parser.add_argument("--r", type=float, default=1.0, help="增长率 r")
parser.add_argument("--alpha", type=float, default=1.0, help="非线性系数 alpha")
parser.add_argument("--nx", type=int, default=1000, help="空间网格点数")
parser.add_argument("--num_steps", type=int, default=1000, help="时间步数")
parser.add_argument("--T", type=float, default=20.0, help="最终时间")
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
r = args.r
alpha = args.alpha

sigma = -r
phi_coeff = r * alpha  # 避免与场变量 phi 冲突，重命名为 phi_coeff
nu = 0.1               # 扩散系数

# 时间参数
T = args.T
num_steps = args.num_steps
dt_val = T / num_steps

# 空间参数
nx = args.nx
x_left = -20.0
x_right = 20.0

save_dir = f"/root/autodl-tmp/fisher/alpha_{alpha:.1f}_r_{r:.1f}"
if rank == 0 and not os.path.exists(save_dir):
    os.makedirs(save_dir)
comm.Barrier()

if rank == 0:
    print(f"参数: σ={sigma}, φ={phi_coeff}, ν={nu}")
    print(f"区域: [{x_left}, {x_right}], 时间: [0, {T}]")
    print(f"网格: {nx}个空间点, {num_steps}个时间步, dt={dt_val:.6e}")

# ---------------------------------------------------------------------------
# 创建网格和函数空间
# ---------------------------------------------------------------------------
domain = mesh.create_interval(comm, nx, [x_left, x_right])
V = fem.functionspace(domain, ("Lagrange", 1))

if rank == 0:
    print(f"Total DOFs: {V.dofmap.index_map.size_global}")

# ---------------------------------------------------------------------------
# 函数定义
# ---------------------------------------------------------------------------
u = fem.Function(V, name="u")          # 当前时间步的解（未知）
u_n = fem.Function(V, name="u_n")      # 上一时间步的解
v = ufl.TestFunction(V)
u_trial = ufl.TrialFunction(V)         # 用于 Jacobian

# ---------------------------------------------------------------------------
# 初始条件: u(x,0) = exp(-x²)
# ---------------------------------------------------------------------------
def initial_condition(x):
    """Vectorized initial condition for DOLFINx interpolation."""
    return np.exp(-x[0] ** 2)

u.interpolate(initial_condition)
u_n.interpolate(initial_condition)

# ---------------------------------------------------------------------------
# 边界条件: u = 0 on ∂Ω (Dirichlet)
# ---------------------------------------------------------------------------
# 找到边界自由度
fdim = domain.topology.dim - 1  # facet dimension = 0 for 1D
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

# ---------------------------------------------------------------------------
# 时间步长（用 dolfinx.fem.Constant）
# ---------------------------------------------------------------------------
dt = fem.Constant(domain, PETSc.ScalarType(dt_val))

# ---------------------------------------------------------------------------
# 变分形式 (残差 F, 自动求导 Jacobian J)
# ---------------------------------------------------------------------------
# PDE: (u - u_n)/dt + σ·u + φ·u² - ν·Δu = 0
# 弱形式: ∫[(u-u_n)/dt·v + σ·u·v + φ·u²·v + ν·∇u·∇v] dx = 0

dx = ufl.dx

F_form = (
    (u - u_n) / dt * v * dx
    + fem.Constant(domain, PETSc.ScalarType(sigma)) * u * v * dx
    + fem.Constant(domain, PETSc.ScalarType(phi_coeff)) * u**2 * v * dx
    + fem.Constant(domain, PETSc.ScalarType(nu)) * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
)

J_form_ufl = ufl.derivative(F_form, u, u_trial)

# ---------------------------------------------------------------------------
# 编译 UFL form 为 dolfinx.fem.Form（用于 BC 施加等 CPU 操作）
# ---------------------------------------------------------------------------
F_compiled = fem.form(F_form)
J_compiled = fem.form(J_form_ufl)

# ---------------------------------------------------------------------------
# cuDOLFINx: 直接用 UFL form 创建 CUDA form
# ---------------------------------------------------------------------------
cuda_F_form = cufem.form(F_form)
cuda_J_form = cufem.form(J_form_ufl)
cuda_asm = cufem.CUDAAssembler()

# ---------------------------------------------------------------------------
# Newton 更新量
# ---------------------------------------------------------------------------
du = fem.Function(V)

# ---------------------------------------------------------------------------
# 配置 PETSc KSP 线性求解器 — 全 GPU 路径
# ---------------------------------------------------------------------------
solver = PETSc.KSP().create(comm)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.JACOBI)
solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
solver.setGMRESRestart(200)
solver.setFromOptions()

if rank == 0:
    print(f"[INFO] KSP type: {solver.getType()}, PC type: {solver.getPC().getType()}")

# ---------------------------------------------------------------------------
# 自定义 Newton 求解器（使用 cuDOLFINx GPU 组装）
# ---------------------------------------------------------------------------
def newton_solve(
    u: fem.Function,
    u_n: fem.Function,
    bcs: list,
    max_iter: int = 25,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    relaxation: float = 1.0,
):
    """
    手动实现 Newton 迭代 + 简易 line search，使用 cuDOLFINx GPU 组装。

    Returns:
        (converged: bool, iterations: int)
    """
    for i in range(max_iter):
        # --- GPU 组装残差向量 ---
        cuda_b = cuda_asm.assemble_vector(cuda_F_form)
        b_vec = cuda_b.vector
        b_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        # 施加 Dirichlet BC 到残差
        dolfinx.fem.petsc.apply_lifting(b_vec, [J_compiled], [bcs])
        dolfinx.fem.petsc.set_bc(b_vec, bcs)

        # 检查残差范数
        res_norm = b_vec.norm(PETSc.NormType.NORM_2)

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
        A_mat = cuda_A.mat

        # 施加 Dirichlet BC 到矩阵
        # 注意: 对于 GPU 矩阵，可能需要在 CPU 上处理 BC
        # dolfinx.fem.petsc.apply_lifting 和 set_bc 已在残差上完成

        # --- 求解 J * du = -F ---
        solver.setOperators(A_mat)

        x_gpu = b_vec.duplicate()
        x_gpu.set(0.0)

        solver.solve(b_vec, x_gpu)

        ksp_its = solver.getIterationNumber()
        ksp_reason = solver.getConvergedReason()
        correction_norm = x_gpu.norm(PETSc.NormType.NORM_2)

        if rank == 0:
            print(
                f"  Newton iter {i}: |F|={res_norm:.6e}, "
                f"|du|={correction_norm:.6e}, KSP its={ksp_its}, reason={ksp_reason}"
            )

        # KSP 没收敛则 Newton 失败
        if ksp_reason < 0:
            if rank == 0:
                print(f"  KSP diverged (reason={ksp_reason}), Newton failed")
            x_gpu.destroy()
            return False, i + 1

        # 将 GPU 解拷贝回 CPU 的 du
        x_gpu_array = x_gpu.getArray()
        du.x.array[: len(x_gpu_array)] = x_gpu_array
        du.x.scatter_forward()
        x_gpu.destroy()

        # --- 简易 line search: 回溯如果残差增大 ---
        alpha_ls = relaxation
        u_backup = u.x.array.copy()

        for ls in range(5):
            u.x.array[:] = u_backup + alpha_ls * du.x.array[:]
            u.x.scatter_forward()

            # 计算新残差
            cuda_b_new = cuda_asm.assemble_vector(cuda_F_form)
            b_new = cuda_b_new.vector
            b_new.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES,
                mode=PETSc.ScatterMode.REVERSE,
            )
            new_res = b_new.norm(PETSc.NormType.NORM_2)

            if new_res < res_norm * 1.5 or alpha_ls < 0.05:
                break
            alpha_ls *= 0.5
            if rank == 0:
                print(f"    line search: alpha={alpha_ls:.4f}, |F_new|={new_res:.6e}")

        # 发散检测
        if not np.isfinite(correction_norm) or correction_norm > 1e15:
            if rank == 0:
                print(f"  Newton diverged at iter {i}")
            return False, i + 1

    return False, max_iter


# ---------------------------------------------------------------------------
# 保存函数
# ---------------------------------------------------------------------------
def save_checkpoint(solutions_data, times_list, coords):
    """保存完整的解数据到 .npy 文件"""
    n_times = len(times_list)
    n_points = len(coords)
    sol_array = np.array(solutions_data)  # (n_times, n_points)

    np.save(f"{save_dir}/mesh.npy", coords)
    np.save(f"{save_dir}/times.npy", np.array(times_list))
    np.save(f"{save_dir}/sol.npy", sol_array)
    np.save(
        f"{save_dir}/params.npy",
        {
            "sigma": sigma,
            "phi": phi_coeff,
            "nu": nu,
            "T": T,
            "num_steps": num_steps,
            "nx": nx,
            "x_left": x_left,
            "x_right": x_right,
        },
    )

    if rank == 0:
        print(f"解数据已保存到 {save_dir}/")
        print(f"解数组形状: {sol_array.shape} (时间点 x 空间点)")


# ---------------------------------------------------------------------------
# 时间推进主循环
# ---------------------------------------------------------------------------
if rank == 0:
    print("\n开始求解一维Fisher方程...")

# 获取网格坐标（DOF 坐标）
coords = V.tabulate_dof_coordinates()[:, 0]  # 1D，只取 x 分量

# 存储解的列表
solutions_data = [u_n.x.array.copy()]
times_list = [0.0]

t = 0.0
save_interval = 10  # 每10步存储一次

for n in range(num_steps):
    t += dt_val

    # 将上一时刻解作为本次 Newton 初猜
    u.x.array[:] = u_n.x.array[:]
    u.x.scatter_forward()

    # Newton 求解
    converged, iterations = newton_solve(
        u, u_n, bcs=[bc], max_iter=25, atol=1e-8, rtol=1e-6, relaxation=1.0
    )

    if not converged:
        if rank == 0:
            print(f"Newton solver failed at t={t:.6f}, step {n+1}/{num_steps}")
        break

    # 更新上一时间步的解
    u_n.x.array[:] = u.x.array[:]
    u_n.x.scatter_forward()

    # 存储解（每 save_interval 步存储一次）
    if (n + 1) % save_interval == 0 or n == num_steps - 1:
        solutions_data.append(u_n.x.array.copy())
        times_list.append(t)

    # 输出进度
    if (n + 1) % 100 == 0 and rank == 0:
        print(f"时间步 {n+1}/{num_steps}, t = {t:.3f}")

# ---------------------------------------------------------------------------
# 保存结果
# ---------------------------------------------------------------------------
if rank == 0:
    save_checkpoint(solutions_data, times_list, coords)

# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------
solver.destroy()

if rank == 0:
    end_time = time.time()
    print(f"\n求解完成! 共保存了 {len(solutions_data)} 个时间点")
    print(f"求解耗时: {end_time - start_time:.2f} 秒")