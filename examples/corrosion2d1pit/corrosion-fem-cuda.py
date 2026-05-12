#!/usr/bin/env python3

import argparse
import json
import math
import os
import time

import basix.ufl
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh
import dolfinx.fem.petsc as fem_petsc

os.environ.setdefault("OMP_NUM_THREADS", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D corrosion model with coupled Allen-Cahn/Cahn-Hilliard dynamics."
    )
    parser.add_argument("--ratio", type=float, default=1.0, help="Scaling ratio for alphap")
    parser.add_argument("--dt", type=float, default=1e-3, help="Initial time step size")
    parser.add_argument("--t-end", type=float, default=30.0, help="Final simulation time")
    parser.add_argument("--nx", type=int, default=150, help="Number of elements in x")
    parser.add_argument("--ny", type=int, default=75, help="Number of elements in y")
    parser.add_argument(
        "--output-every",
        type=float,
        default=0.1,
        help="Write checkpoints every this much simulation time",
    )
    parser.add_argument(
        "--max-newton-iters",
        type=int,
        default=30,
        help="Maximum Newton iterations per time step",
    )
    parser.add_argument(
        "--debug-first-step",
        action="store_true",
        help="Write detailed diagnostics for the first accepted time step",
    )
    parser.add_argument(
        "--ch-scale",
        type=float,
        default=1e6,
        help="Scaling factor applied to the Cahn-Hilliard residual block",
    )
    parser.add_argument(
        "--pc-strategy",
        type=str,
        default="jacobi",
        choices=["jacobi", "fieldsplit"],
        help="Linear preconditioning strategy for the PETSc KSP solver",
    )
    parser.add_argument(
        "--profile-solver",
        action="store_true",
        help="Collect timing and iteration statistics for assembly, KSP, and line search",
    )
    parser.add_argument(
        "--jacobian-lag",
        type=int,
        default=1,
        help="Assemble the Jacobian every N Newton iterations (1 means every iteration)",
    )
    parser.add_argument(
        "--max-line-search-steps",
        type=int,
        default=3,
        help="Maximum backtracking steps in the line search",
    )
    parser.add_argument(
        "--sub-ksp-type",
        type=str,
        default="cg",
        choices=["cg", "gmres"],
        help="KSP type for fieldsplit sub-solvers",
    )
    parser.add_argument(
        "--sub-pc-type",
        type=str,
        default="jacobi",
        choices=["jacobi", "none"],
        help="PC type for fieldsplit sub-solvers",
    )
    args, petsc_args = parser.parse_known_args()
    PETSc.Options().insertString(" ".join(petsc_args))
    return args


def ratio_tag(value):
    return f"{value:g}".replace("+", "")


def gather_on_root(comm, local_array):
    chunks = comm.gather(local_array, root=0)
    if comm.rank != 0:
        return None
    nonempty = [chunk for chunk in chunks if chunk.size > 0]
    if not nonempty:
        return np.empty((0,), dtype=local_array.dtype)
    return np.concatenate(nonempty, axis=0)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = time.time()

    alphap = 1.03e-4 / args.ratio
    omegap = 1.76e7
    DD = 8.5e-10
    AA = 5.35e7
    Lp = 2.0
    cse = 1.0
    cle = 5100.0 / 1.43e5

    pit_radius = 5e-6
    domain_min = np.array([-50e-6, 0.0], dtype=np.float64)
    domain_max = np.array([50e-6, 50e-6], dtype=np.float64)
    domain_area = (domain_max[0] - domain_min[0]) * (domain_max[1] - domain_min[1])

    # save_dir = os.path.join(os.getcwd(), "results_cuda", f"ratio-{ratio_tag(args.ratio)}")
    save_dir = f"/root/autodl-tmp/corrosion/ratio-{ratio_tag(args.ratio)}-pc_{args.pc_strategy}"
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    comm.Barrier()

    domain = mesh.create_rectangle(
        comm,
        [domain_min, domain_max],
        [args.nx, args.ny],
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.crossed,
    )

    scalar_element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    mixed_element = basix.ufl.mixed_element([scalar_element, scalar_element])
    W = fem.functionspace(domain, mixed_element)

    if rank == 0:
        total_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
        print(f"Total mixed DOFs: {total_dofs}")

    pc = fem.Function(W, name="pc")
    pc_n = fem.Function(W, name="pc_n")
    p, c = ufl.split(pc)
    p_n, c_n = ufl.split(pc_n)
    q_p, q_c = ufl.TestFunctions(W)
    trial = ufl.TrialFunction(W)

    Vp, p_to_W = W.sub(0).collapse()
    Vc, c_to_W = W.sub(1).collapse()
    p_to_W = np.asarray(p_to_W, dtype=np.int32)
    c_to_W = np.asarray(c_to_W, dtype=np.int32)
    p_out = fem.Function(Vp, name="p")
    c_out = fem.Function(Vc, name="c")

    mesh_points_local = Vp.tabulate_dof_coordinates()[:, : domain.geometry.dim].copy()
    mesh_points = gather_on_root(comm, mesh_points_local)
    if rank == 0:
        np.save(os.path.join(save_dir, "mesh_points.npy"), mesh_points)

    interface_scale = math.sqrt(omegap / (2.0 * alphap))

    def initial_phase(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return 0.5 * (1.0 + np.tanh(interface_scale * (r - pit_radius)))

    def initial_concentration(x):
        phase = initial_phase(x)
        h_phase = -2.0 * phase**3 + 3.0 * phase**2
        return h_phase * cse

    p_init = fem.Function(Vp)
    c_init = fem.Function(Vc)
    p_init.interpolate(initial_phase)
    c_init.interpolate(initial_concentration)
    pc.x.array[p_to_W] = p_init.x.array
    pc.x.array[c_to_W] = c_init.x.array
    pc.x.scatter_forward()
    pc_n.x.array[:] = pc.x.array
    pc_n.x.scatter_forward()

    def on_pit_boundary(x):
        return np.isclose(x[1], 0.0) & (x[0] ** 2 + x[1] ** 2 <= pit_radius**2)

    zero_p = fem.Function(Vp)
    zero_c = fem.Function(Vc)
    zero_p.x.array[:] = 0.0
    zero_c.x.array[:] = 0.0

    p_dofs_sub = fem.locate_dofs_geometrical((W.sub(0), Vp), on_pit_boundary)
    c_dofs_sub = fem.locate_dofs_geometrical((W.sub(1), Vc), on_pit_boundary)
    p_dofs_local = fem.locate_dofs_geometrical(Vp, on_pit_boundary)
    c_dofs_local = fem.locate_dofs_geometrical(Vc, on_pit_boundary)
    p_dofs_parent = p_to_W[p_dofs_local]
    c_dofs_parent = c_to_W[c_dofs_local]

    bcs = [
        fem.dirichletbc(zero_p, p_dofs_sub, W.sub(0)),
        fem.dirichletbc(zero_c, c_dofs_sub, W.sub(1)),
    ]

    def subspace_dof_count(dofs):
        if isinstance(dofs, (list, tuple)):
            return len(dofs[0])
        return len(dofs)

    debug_report = None
    if args.debug_first_step:
        debug_report = {
            "ratio": float(args.ratio),
            "initial_dt": float(args.dt),
            "ch_scale": float(args.ch_scale),
            "p_dirichlet_dofs": int(subspace_dof_count(p_dofs_sub)),
            "c_dirichlet_dofs": int(subspace_dof_count(c_dofs_sub)),
            "p_parent_dirichlet_dofs": int(len(p_dofs_parent)),
            "c_parent_dirichlet_dofs": int(len(c_dofs_parent)),
        }

    dt = fem.Constant(domain, PETSc.ScalarType(args.dt))
    dx = ufl.dx

    h_p = -2.0 * p**3 + 3.0 * p**2
    dh_dp = -6.0 * p**2 + 6.0 * p
    g_p = p**2 * (1.0 - p) ** 2

    F_ch = (
        (c - c_n) / dt * q_c * dx
        - ufl.inner(-DD * ufl.grad(c) + DD * (cse - cle) * ufl.grad(h_p), ufl.grad(q_c)) * dx
    )
    F_ac = (
        (p - p_n) / (Lp * dt) * q_p * dx
        - 2.0 * AA * (c - h_p * (cse - cle) - cle) * (cse - cle) * dh_dp * q_p * dx
        + omegap * (4.0 * p**3 - 6.0 * p**2 + 2.0 * p) * q_p * dx
        + ufl.inner(alphap * ufl.grad(p), ufl.grad(q_p)) * dx
    )
    F = args.ch_scale * F_ch + F_ac
    J = ufl.derivative(F, pc, trial)

    residual_form = fem.form(F)
    jacobian_form = fem.form(J)
    free_energy_form = fem.form(
        (
            AA * (c - h_p * (cse - cle) - cle) ** 2
            + omegap * g_p
            + 0.5 * alphap * ufl.inner(ufl.grad(p), ufl.grad(p))
        )
        * dx
    )
    concentration_form = fem.form((2.0 * c + (1.0 - 2.0 * h_p) * (cse - cle)) * dx)

    A = fem_petsc.create_matrix(jacobian_form)
    try:
        A.setType(PETSc.Mat.Type.AIJCUSPARSE)
    except AttributeError:
        A.setType("aijcusparse")
    A.setUp()
    b = fem_petsc.create_vector(residual_form)

    solver = PETSc.KSP().create(comm)
    linear_correction_vec = None
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setGMRESRestart(200)

    ksp_pc = solver.getPC()
    pc_strategy = args.pc_strategy.lower()
    split_is = None
    if pc_strategy == "fieldsplit":
        ksp_pc.setType(PETSc.PC.Type.FIELDSPLIT)
        ksp_pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        p_is = PETSc.IS().createGeneral(p_to_W.astype(np.int32), comm=comm)
        c_is = PETSc.IS().createGeneral(c_to_W.astype(np.int32), comm=comm)
        ksp_pc.setFieldSplitIS(("p", p_is), ("c", c_is))
        split_is = (p_is, c_is)
    else:
        ksp_pc.setType(PETSc.PC.Type.JACOBI)

    solver.setFromOptions()

    if rank == 0:
        print(f"KSP type: {solver.getType()}, PC type: {solver.getPC().getType()}")
        print(f"Matrix type: {A.getType()}")
        print(f"CH residual scale: {args.ch_scale:.6e}")
        print(f"PC strategy: {pc_strategy}")
        print(f"Jacobian lag: {args.jacobian_lag}")
        print(f"Max line-search steps: {args.max_line_search_steps}")
        print(f"Sub KSP type: {args.sub_ksp_type}")
        print(f"Sub PC type: {args.sub_pc_type}")

    correction = fem.Function(W)
    linear_correction_vec = A.createVecRight()

    def enforce_state_bcs(function):
        function.x.array[p_dofs_parent] = 0.0
        function.x.array[c_dofs_parent] = 0.0
        function.x.scatter_forward()

    enforce_state_bcs(pc)
    enforce_state_bcs(pc_n)

    def update_scalar_views():
        p_out.x.array[:] = pc.x.array[p_to_W]
        c_out.x.array[:] = pc.x.array[c_to_W]
        p_out.x.scatter_forward()
        c_out.x.scatter_forward()

    def compute_free_energy():
        local_value = fem.assemble_scalar(free_energy_form)
        return comm.allreduce(local_value, op=MPI.SUM)

    def compute_concentration_conservation():
        local_value = fem.assemble_scalar(concentration_form)
        return comm.allreduce(local_value, op=MPI.SUM) / domain_area

    xdmf = io.XDMFFile(comm, os.path.join(save_dir, "fields.xdmf"), "w")
    xdmf.write_mesh(domain)

    last_saved_time = [None]

    def save_checkpoint(current_time):
        if last_saved_time[0] is not None and abs(current_time - last_saved_time[0]) < 1e-12:
            return

        update_scalar_views()
        xdmf.write_function(p_out, current_time)
        xdmf.write_function(c_out, current_time)

        coords = gather_on_root(comm, mesh_points_local)
        p_values = gather_on_root(comm, p_out.x.array.copy())
        c_values = gather_on_root(comm, c_out.x.array.copy())
        last_saved_time[0] = current_time
        if rank == 0:
            np.savez(
                os.path.join(save_dir, f"sol-{current_time:.4f}.npz"),
                coords=coords,
                p=p_values,
                c=c_values,
            )

    def write_diagnostics_header():
        if rank == 0:
            with open(os.path.join(save_dir, "time_logs.txt"), "w", encoding="utf-8") as handle:
                handle.write("sim_time,compute_time,dt,free_energy,concentration_conservation\n")

    def append_diagnostics(current_time, current_dt, free_energy, concentration_conservation):
        if rank == 0:
            with open(os.path.join(save_dir, "time_logs.txt"), "a", encoding="utf-8") as handle:
                handle.write(
                    f"{current_time},{time.time() - start_time},{current_dt},{free_energy},{concentration_conservation}\n"
                )

    def assemble_residual():
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem_petsc.assemble_vector(b, residual_form)
        fem_petsc.apply_lifting(b, [jacobian_form], bcs=[bcs], x0=[pc.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs, pc.x.petsc_vec, -1.0)
        return b.norm(PETSc.NormType.NORM_2)

    def assemble_jacobian():
        A.zeroEntries()
        fem_petsc.assemble_matrix(A, jacobian_form, bcs=bcs)
        A.assemble()

    def newton_solve(max_iter, atol=1e-8, rtol=1e-6, relaxation=1.0):
        residual_0 = None
        residual_norm = None
        correction_0 = None
        correction_norm = None
        line_search_alphas = []
        residual_is_fresh = False
        timing = {
            "assemble_residual": 0.0,
            "assemble_jacobian": 0.0,
            "linear_solve": 0.0,
            "line_search": 0.0,
        }
        total_ksp_iterations = 0
        total_sub_ksp_iterations = []
        line_search_evaluations = 0

        for iteration in range(max_iter):
            if not residual_is_fresh:
                tic = time.perf_counter()
                residual_norm = assemble_residual()
                timing["assemble_residual"] += time.perf_counter() - tic
            residual_is_fresh = False
            if iteration == 0:
                residual_0 = residual_norm if residual_norm > 0 else 1.0

            b.scale(-1.0)
            needs_new_jacobian = iteration == 0 or max(1, args.jacobian_lag) == 1 or iteration % max(1, args.jacobian_lag) == 0
            if needs_new_jacobian:
                tic = time.perf_counter()
                assemble_jacobian()
                timing["assemble_jacobian"] += time.perf_counter() - tic
                solver.setOperators(A)
                if pc_strategy == "fieldsplit" and iteration == 0:
                    solver.setUp()
                    sub_ksps = solver.getPC().getFieldSplitSubKSP()
                    for sub_ksp in sub_ksps:
                        if args.sub_ksp_type == "cg":
                            sub_ksp.setType(PETSc.KSP.Type.CG)
                        else:
                            sub_ksp.setType(PETSc.KSP.Type.GMRES)
                            sub_ksp.setGMRESRestart(100)
                        if args.sub_pc_type == "none":
                            sub_ksp.getPC().setType(PETSc.PC.Type.NONE)
                        else:
                            sub_ksp.getPC().setType(PETSc.PC.Type.JACOBI)
                        sub_ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)

            linear_correction_vec.set(0.0)
            tic = time.perf_counter()
            solver.solve(b, linear_correction_vec)
            timing["linear_solve"] += time.perf_counter() - tic

            ksp_reason = solver.getConvergedReason()
            ksp_iterations = solver.getIterationNumber()
            total_ksp_iterations += ksp_iterations
            if pc_strategy == "fieldsplit":
                sub_ksps = solver.getPC().getFieldSplitSubKSP()
                total_sub_ksp_iterations.extend(sub_ksp.getIterationNumber() for sub_ksp in sub_ksps)
            correction_norm = linear_correction_vec.norm(PETSc.NormType.NORM_2)
            if iteration == 0:
                correction_0 = correction_norm if correction_norm > 0 else 1.0
            correction_array = linear_correction_vec.getArray(readonly=True)
            correction.x.array[:] = correction_array
            correction.x.scatter_forward()

            if rank == 0:
                print(
                    f"  Newton iter {iteration}: |F|={residual_norm:.6e}, "
                    f"|du|={correction_norm:.6e}, KSP its={ksp_iterations}, reason={ksp_reason}"
                )

            if ksp_reason < 0 or not np.isfinite(correction_norm) or correction_norm > 1e15:
                return False, iteration + 1, {
                    "initial_residual": float(residual_0),
                    "final_residual": float(residual_norm),
                    "initial_correction": float(correction_0 if correction_0 is not None else 0.0),
                    "final_correction": float(correction_norm),
                    "line_search_alphas": line_search_alphas,
                    "timing": timing,
                    "total_ksp_iterations": int(total_ksp_iterations),
                    "sub_ksp_iterations": [int(v) for v in total_sub_ksp_iterations],
                    "line_search_evaluations": int(line_search_evaluations),
                    "jacobian_lag": int(max(1, args.jacobian_lag)),
                    "max_line_search_steps": int(max(1, args.max_line_search_steps)),
                }

            state_backup = pc.x.array.copy()
            alpha = relaxation
            accepted_residual = residual_norm
            tic = time.perf_counter()
            for _ in range(max(1, args.max_line_search_steps)):
                pc.x.array[:] = state_backup + alpha * correction.x.array
                enforce_state_bcs(pc)

                trial_norm = assemble_residual()
                line_search_evaluations += 1
                if trial_norm < residual_norm or alpha < 0.02:
                    accepted_residual = trial_norm
                    residual_is_fresh = True
                    line_search_alphas.append(float(alpha))
                    break
                alpha *= 0.5
                if rank == 0:
                    print(f"    line search: alpha={alpha:.4f}, |F_new|={trial_norm:.6e}")
            timing["line_search"] += time.perf_counter() - tic

            residual_norm = accepted_residual
            if correction_norm < atol or correction_norm / correction_0 < rtol:
                if rank == 0:
                    print(
                        f"  Newton iter {iteration}: |du|={correction_norm:.6e}, "
                        f"|F|={residual_norm:.6e} -> converged"
                    )
                return True, iteration + 1, {
                    "initial_residual": float(residual_0),
                    "final_residual": float(residual_norm),
                    "initial_correction": float(correction_0),
                    "final_correction": float(correction_norm),
                    "line_search_alphas": line_search_alphas,
                    "timing": timing,
                    "total_ksp_iterations": int(total_ksp_iterations),
                    "sub_ksp_iterations": [int(v) for v in total_sub_ksp_iterations],
                    "line_search_evaluations": int(line_search_evaluations),
                    "jacobian_lag": int(max(1, args.jacobian_lag)),
                    "max_line_search_steps": int(max(1, args.max_line_search_steps)),
                }

        return False, max_iter, {
            "initial_residual": float(residual_0 if residual_0 is not None else 0.0),
            "final_residual": float(residual_norm if residual_norm is not None else 0.0),
            "initial_correction": float(correction_0 if correction_0 is not None else 0.0),
            "final_correction": float(correction_norm if correction_norm is not None else 0.0),
            "line_search_alphas": line_search_alphas,
            "timing": timing,
            "total_ksp_iterations": int(total_ksp_iterations),
            "line_search_evaluations": int(line_search_evaluations),
        }

    current_time = 0.0
    current_dt = args.dt
    dt_max = 0.2
    dt_min = 1e-5
    next_output_time = args.output_every
    previous_free_energy = None
    previous_concentration = None

    write_diagnostics_header()
    save_checkpoint(current_time)
    initial_free_energy = compute_free_energy()
    initial_concentration = compute_concentration_conservation()
    append_diagnostics(current_time, current_dt, initial_free_energy, initial_concentration)
    previous_free_energy = initial_free_energy
    previous_concentration = initial_concentration

    if rank == 0:
        print(f"Initial free energy: {initial_free_energy:.6e}")
        print(f"Initial concentration conservation: {initial_concentration:.6e}")

    while current_time < args.t_end:
        step_dt = min(current_dt, args.t_end - current_time)
        dt.value = step_dt
        pc.x.array[:] = pc_n.x.array
        enforce_state_bcs(pc)
        p_before_step = pc.x.array[p_to_W].copy()
        c_before_step = pc.x.array[c_to_W].copy()

        converged, iterations, step_debug = newton_solve(
            args.max_newton_iters,
            atol=1e-8,
            rtol=1e-6,
            relaxation=0.9,
        )
        if not converged:
            if rank == 0:
                print(
                    f"Newton solver failed at t={current_time:.6f}. "
                    f"Decreasing dt from {step_dt:.6e} to {step_dt / 2.0:.6e}"
                )
            current_dt = step_dt / 2.0
            if current_dt < dt_min:
                if rank == 0:
                    print("Time step reached minimum limit. Stopping simulation.")
                break
            continue

        current_time += step_dt
        enforce_state_bcs(pc)
        p_after_step = pc.x.array[p_to_W].copy()
        c_after_step = pc.x.array[c_to_W].copy()
        if debug_report is not None and "first_step" not in debug_report:
            dp = p_after_step - p_before_step
            dc = c_after_step - c_before_step
            debug_report["first_step"] = {
                "accepted_time": float(current_time),
                "dt": float(step_dt),
                "newton_iterations": int(iterations),
                "initial_residual": float(step_debug["initial_residual"]),
                "final_residual": float(step_debug["final_residual"]),
                "residual_ratio": float(step_debug["final_residual"] / step_debug["initial_residual"]) if step_debug["initial_residual"] else 0.0,
                "initial_correction": float(step_debug["initial_correction"]),
                "final_correction": float(step_debug["final_correction"]),
                "correction_ratio": float(step_debug["final_correction"] / step_debug["initial_correction"]) if step_debug["initial_correction"] else 0.0,
                "line_search_alphas": step_debug["line_search_alphas"],
                "timing": step_debug["timing"],
                "total_ksp_iterations": int(step_debug["total_ksp_iterations"]),
                "line_search_evaluations": int(step_debug["line_search_evaluations"]),
                "dp_norm": float(np.linalg.norm(dp)),
                "dc_norm": float(np.linalg.norm(dc)),
                "dp_max_abs": float(np.max(np.abs(dp))),
                "dc_max_abs": float(np.max(np.abs(dc))),
                "changed_p_dofs": int(np.count_nonzero(np.abs(dp) > 1e-14)),
                "changed_c_dofs": int(np.count_nonzero(np.abs(dc) > 1e-14)),
            }
            if rank == 0:
                with open(os.path.join(save_dir, "debug_first_step.json"), "w", encoding="utf-8") as handle:
                    json.dump(debug_report, handle, indent=2)
                np.savez(
                    os.path.join(save_dir, "debug_first_step_fields.npz"),
                    coords=mesh_points,
                    p_before=p_before_step,
                    p_after=p_after_step,
                    c_before=c_before_step,
                    c_after=c_after_step,
                    dp=dp,
                    dc=dc,
                )
                print(f"Debug report written to {os.path.join(save_dir, 'debug_first_step.json')}")
        pc_n.x.array[:] = pc.x.array
        enforce_state_bcs(pc_n)

        free_energy = compute_free_energy()
        concentration_conservation = compute_concentration_conservation()
        append_diagnostics(current_time, step_dt, free_energy, concentration_conservation)

        if rank == 0:
            print(f"Iterations: {iterations}. Step: {step_dt:.6e}. Time: {current_time:.6f}")
            if args.profile_solver:
                timing = step_debug["timing"]
                print(
                    "  Solver profile: "
                    f"residual={timing['assemble_residual']:.3e}s, "
                    f"jacobian={timing['assemble_jacobian']:.3e}s, "
                    f"linear={timing['linear_solve']:.3e}s, "
                    f"line_search={timing['line_search']:.3e}s, "
                    f"ksp_total={step_debug['total_ksp_iterations']}, "
                    f"sub_ksp={step_debug.get('sub_ksp_iterations', [])}, "
                    f"ls_evals={step_debug['line_search_evaluations']}, "
                    f"jac_lag={step_debug['jacobian_lag']}, "
                    f"ls_max={step_debug['max_line_search_steps']}"
                )
            print(
                f"Free energy: {free_energy:.6e}, "
                f"Change: {free_energy - previous_free_energy:.6e}"
            )
            print(
                f"Concentration conservation: {concentration_conservation:.6e}, "
                f"Change: {concentration_conservation - previous_concentration:.6e}"
            )

        previous_free_energy = free_energy
        previous_concentration = concentration_conservation

        if iterations < 10 and current_dt < dt_max:
            current_dt = min(2.0*current_dt, dt_max)
            if rank == 0:
                print(f"Increasing dt to {current_dt:.6e}")
        else:
            current_dt = step_dt

        if current_time + 1e-12 >= next_output_time:
            save_checkpoint(current_time)
            next_output_time += args.output_every

    save_checkpoint(current_time)
    xdmf.close()
    solver.destroy()
    linear_correction_vec.destroy()
    if split_is is not None:
        for iset in split_is:
            iset.destroy()
    A.destroy()
    b.destroy()

    if rank == 0:
        end_time = time.time()
        print("Simulation finished")
        print(f"Time elapsed: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()
"""Reference solution of 1D freely-propagating premixed flames based on simplified physical models."""
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 22
set_dpi = 200
plt.rcParams["font.size"] = set_fs  # default font size
# plt.rcParams["font.sans-serif"] = "Arial"  # default font (for Windows)
# plt.rcParams["font.sans-serif"] = "Nimbus Sans"  # default font (for Linux)
# plt.rcParams["font.sans-serif"] = "Times New Roman"  # default font
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


start_time = time.time()
# ----------------------------------------------------------------------
# define the constants
W = 28.97e-3  # gas molecular weight, kg/mol
lam = 2.6e-2  # thermal conductivity, W/(m-K)
cp = 1000.0  # heat capacity, J/(kg-K)
qF = 5.0e7  # fuel calorific value, J/kg

R = 8.3145  # universal gas constant, J/(mol-K)
A = 1.4e8  # pre-exponential factor
Ea = 1.214172e5  # activation energy, J/mol
nu_rxn = 1.6  # reaction order

# Rg = 287  # gas constant, J/(kg-K)
Rg = R / W  # gas constant, J/(kg-K)

# ----------------------------------------------------------------------
# set calculation domain
L = 0.0015  # 1.5 mm
n_grids = 10000  # 10, 50, 100, 1000, 10000, 100000, 1000000
n_steps = 500  # maximum iteration steps
dx = L / (n_grids - 1)
x = np.linspace(0, L, n_grids)
# temperature, temperature gradient, velocity, density, pressure, mass fraction of fuel, reaction rate
T, gradT, u, rho, p, YF, omega = np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids), \
                                  np.zeros(n_grids), np.zeros(n_grids), np.zeros(n_grids)

# ----------------------------------------------------------------------
# inlet boundary condition
T[0] = 400  # K
gradT[0] = 1e5  # K/m
p[0] = 101325 * 1.0  # Pa
phi = 0.40
YF[0] = phi / (phi + (2 * 32 / 16))
rho[0] = p[0] / (Rg * T[0])  # kg/m3
omega[0] = A * np.exp(-Ea / (R * T[0])) * (YF[0] * rho[0]) ** nu_rxn  # kg/m3-s

T_max = T[0] + qF * YF[0] / cp

# save_dir = "./results/p{:.2f}_phi{:.2f}/".format(p[0]/101325, phi)
save_dir = "./results/p{:.2f}_T{:.0f}_phi{:.2f}/".format(p[0]/101325, T[0], phi)
os.makedirs(save_dir + "data/", exist_ok=True)
os.makedirs(save_dir + "pics/", exist_ok=True)

# ----------------------------------------------------------------------
# solve the problem using the bisection method
u0_l = 0.
u0_r = 1.

t0 = time.perf_counter()
for k_u in range(n_steps):
    print("\nk_u: {:d}, u0_r-u0_l = {:.4e}".format(k_u, u0_r - u0_l))
    u[0] = (u0_l + u0_r) / 2
    c1 = dx * rho[0] * cp / lam * u[0]
    c2 = dx * qF / lam
    c3 = u[0] + Rg * T[0] / u[0]
    is_converge = True
    for i in range(1, n_grids):
        gradT[i] = gradT[i - 1] + c1 * gradT[i - 1] - c2 * omega[i - 1]
        T[i] = T[i - 1] + dx * gradT[i]
        if gradT[i] < 0:  # flame flashback, indicating a small u0
            u0_l = u[0]
            is_converge = False
            print("flame flashback, u0 too small, i=", i)
            break
        elif T[i] > T_max:  # flame blows out, indicating a large u0
            u0_r = u[0]
            is_converge = False
            print("flame blows out, u0 too large, i=", i)
            break
        else:
            u[i] = 0.5 * (c3 - np.sqrt(c3 ** 2 - 4 * Rg * T[i]))  # choose the smaller root (subsonic)
            rho[i] = rho[0] * u[0] / u[i]
            p[i] = rho[i] * Rg * T[i]
            # p[i] = p[0] - rho[0] * u[0] * (u[i] - u[0])  # same as the last line
            YF[i] = YF[0] + cp * (T[0] - T[i]) / qF
            # YF[i] = cp * (T_max - T[i]) / qF  # same as the last line
            omega[i] = A * np.exp(-Ea / (R * T[i])) * (YF[i] * rho[i]) ** nu_rxn
            # print("flame propagates, i=", i)

    if is_converge or u0_r - u0_l < 1e-16:  # the result is sensitive to this criterion
        if i < n_grids - 1:
            print("converged, i=", i)
            T[i:] = T[i - 1]
            gradT[i:] = 0.0
            u[i:], rho[i:], p[i:], YF[i:], omega[i:] = u[i - 1], rho[i - 1], p[i - 1], YF[i - 1], omega[i - 1]
        break

time_cal = time.perf_counter() - t0

print(f"\nsL: {u0_l} m/s")

# ----------------------------------------------------------------------
# plot the fields
fields = [T, YF, u, rho, omega, p - p[0], gradT]
mathnames = ["$T$", "$Y_F$", "$u$", r"$\rho$", r"$\omega$", "$p_{rel}$", r"$\nabla{T}$"]
textnames = ["T", "YF", "u", "rho", "omega", "p", "gradT"]
units = ["K", " ", "m/s", "kg/m$^3$", "kg/(m$^3$·s)", "Pa", "K/m"]

for i in range(len(fields)):
    plt.figure(figsize=(8, 6))
    plt.title(mathnames[i])
    plt.xlabel("$x$/mm")
    plt.ylabel(units[i])
    plt.plot(x * 1e3, fields[i], lw=3)
    plt.savefig(save_dir + f"pics/{i+1}_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)  # .png  .svg
    plt.close()

# ----------------------------------------------------------------------
# save the data
np.save(save_dir + "data/x.npy", x)
np.save(save_dir + "data/T.npy", T)
np.save(save_dir + "data/YF.npy", YF)
np.save(save_dir + "data/u.npy", u)
np.save(save_dir + "data/rho.npy", rho)
np.save(save_dir + "data/omega.npy", omega)
np.save(save_dir + "data/p.npy", p)
np.save(save_dir + "data/gradT.npy", gradT)
np.save(save_dir + "data/sL.npy", u[0])

print(T_max)
print(gradT[-10:])

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.4f} s")