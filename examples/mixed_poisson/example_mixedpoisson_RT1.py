import numpy as np
import matplotlib.pyplot as plt

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace, MixedFunctionSpace
from fem_engine.element import QuadRT1, Quad1Dc # <<< USING Discontinuous Q1
from fem_engine.assembler import MixedAssembler, assemble_scalar
from fem_engine.bcs import DirichletBC
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.postprocess import export_vtu

def deform_mesh(mesh, L, W, n_x, n_y):
    hx = L / n_x
    hy = W / n_y
    amplitude = 0.25

    for i, pt in enumerate(mesh.points):
        x, y = pt
        if (x > 1e-6 and x < L - 1e-6) and (y > 1e-6 and y < W - 1e-6):
            dx = amplitude * hx * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / W)
            dy = amplitude * hy * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * y / W)

            mesh.points[i, 0] += dx
            mesh.points[i, 1] += dy


# 1. Exact Analytical Solutions
def exact_pressure(x, y):
    """u = cos(pi*x) * cos(pi*y)"""
    return np.cos(np.pi * x) * np.cos(np.pi * y)

def exact_flux(x, y):
    """q = -grad(u)"""
    qx = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    qy = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return np.array([qx, qy])

def exact_div(x, y):
    """div(q) = -Delta(u)"""
    return 2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)


# 2. Error Integrands for assemble_scalar
def l2_pressure_error(u_gp, grad_u, pos_gp, e):
    x, y = pos_gp
    u_ex = exact_pressure(x, y)
    return (u_gp - u_ex)**2


def l2_flux_error(q_gp, grad_q, pos_gp, e):
    x, y = pos_gp
    q_ex = exact_flux(x, y)
    q_vec = np.array(q_gp).flatten()
    return np.sum((q_vec - q_ex)**2)


# 3. Main Convergence
def run_convergence_study():

    mesh_sizes = [4, 8, 16, 32]

    errors_u = []
    errors_q = []

    L, W = 1.0, 1.0

    print("Starting Convergence Study (Mixed Poisson RT1/Q1dc)...")
    print(f"{'Mesh':<10} | {'L2 Error (u)':<18} | {'L2 Error (q)':<18}")

    for n in mesh_sizes:
        # Mesh
        mesh = create_rectangle_mesh(L, W, n_x=n + 1, n_y=n + 1, x0=0.0, y0=0.0)
        #deform_mesh(mesh, L, W, n, n)

        # Spaces (RT1 + Q1dc)
        V_q = FunctionSpace(mesh, QuadRT1(), n_components=1)
        V_u = FunctionSpace(mesh, Quad1Dc(), n_components=1)

        V = MixedFunctionSpace([V_q, V_u])

        # Weak form
        def mixed_poisson_weak(mapped, pos_gp, e):

            (N_q, B_q_div, q, div_q) = mapped[0]
            (N_u, B_u, u, grad_u) = mapped[1]

            x, y = pos_gp
            f = exact_div(x, y)

            R_q = (N_q @ q).flatten() - B_q_div * u[0]
            R_u = N_u * (div_q[0] - f)

            return [R_q, R_u]

        assembler = MixedAssembler(V, mixed_poisson_weak, quad_degree=5)

        # Boundary Conditions
        def boundary_marker(x, y):
            tol = 1e-6
            return abs(x) < tol or abs(x - L) < tol or abs(y) < tol or abs(y - W) < tol

        def origin_cell_marker(x, y):
            return (x < L/n) and (y < W/n)

        bc_q = DirichletBC(V_q, value=0.0, boundary_marker_func=boundary_marker, component=0)
        bc_u = DirichletBC(V_u, value=exact_pressure, boundary_marker_func=origin_cell_marker, component=0)

        def apply_bcs(R, K, U):
            R, K = bc_q.apply(R, K, U, offset=0, method="strong", is_linear=False)
            R, K = bc_u.apply(R, K, U, offset=V_q.ndofs, method="strong", is_linear=False)
            return R, K

        # Solve
        U0 = np.zeros(V.ndofs)
        U_sol = solve_newton_raphson(U0, assembler.assemble, apply_bcs)

        # Split solution into flux and pressure
        q_sol, u_sol = V.split(U_sol)

        # We export each field to its own file.
        # n_vis_pts=4 will slice every quad into 9 sub-quads for high-res plotting
        export_vtu(V_q, q_sol, f"results/flux_RT1_{n}.vtu", field_name="Flux", n_vis_pts=4)
        export_vtu(V_u, u_sol, f"results/pressure_RT1_{n}.vtu", field_name="Pressure", n_vis_pts=4)

        # Export the error fields directly
        export_vtu(V_u, u_sol, f"results/pressure_RT1_err_{n}.vtu", field_name="Pressure_Error", n_vis_pts=4, exact_func=exact_pressure)
        export_vtu(V_q, q_sol, f"results/flux_RT1_err_{n}.vtu", field_name="Flux_Error", n_vis_pts=None, exact_func=exact_flux)

        # Calculate Errors
        err_u_sq = assemble_scalar(V_u, l2_pressure_error, u_sol=u_sol, quad_degree=5)
        err_u = np.sqrt(err_u_sq)

        err_q_sq = assemble_scalar(V_q, l2_flux_error, u_sol=q_sol, quad_degree=5)
        err_q = np.sqrt(err_q_sq)

        errors_u.append(err_u)
        errors_q.append(err_q)

        print(f"{n:>2}x{n:<6} | {err_u:<18.6e} | {err_q:<18.6e}")

    # Convergence rates
    print("\nConvergence Rates:")

    for i in range(1, len(mesh_sizes)):
        rate_u = np.log2(errors_u[i-1] / errors_u[i])
        rate_q = np.log2(errors_q[i-1] / errors_q[i])

        print(f"{mesh_sizes[i-1]} -> {mesh_sizes[i]} : "f"u = {rate_u:.4f}, q = {rate_q:.4f}")


if __name__ == "__main__":
    run_convergence_study()