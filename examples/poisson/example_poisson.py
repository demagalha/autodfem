import numpy as np

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace
from fem_engine.bcs import DirichletBC
from fem_engine.assembler import Assembler, assemble_scalar
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.element import Quad4
from fem_engine.postprocess import export_vtu

# Problem Definition & Exact Solution
L, W = 1.0, 1.0

def u_exact(x, y):
    return 5 + np.sin(np.pi * x) * np.sin(np.pi * y)

def f_source(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_solution(x, y):
    return [u_exact(x, y)]

# The Generic Weak Form Callback
def poisson_weak_form(N, B_x, u_gp, grad_u, x_gp, e):
    """
    Evaluates: (grad_v * grad_u) - (v * f)
    For a scalar PDE, grad_u is shape (2, 1). B_x is shape (2, 4).
    """
    f_val = f_source(x_gp[0], x_gp[1])
    
    # Diffusion term: B_x.T * grad_u
    diff_term = B_x.T @ grad_u
    
    # Source term: N * f
    # so outer product results in shape (4, 1) matching diff_term.
    source_term = -np.outer(N, [f_val])
    
    return diff_term + source_term

if __name__ == "__main__":
    # Setup Generic Framework for Scalar PDE
    mesh = create_rectangle_mesh(L=L, W=W, n_x=30, n_y=30, x0=0.0, y0=0.0)

    # n_components=1 because Poisson is a scalar field
    V = FunctionSpace(mesh, Quad4(), n_components=1)

    assembler = Assembler(V, poisson_weak_form, quad_degree=2)

    # Setup Boundary Conditions
    def boundary_marker(x, y):
        tol = 1e-6
        return (abs(x) < tol or abs(x - L) < tol or 
                abs(y) < tol or abs(y - W) < tol)

    # Pin the entire boundary to 5.0 for component 0 (the only component)
    bc = DirichletBC(V, value=5.0, boundary_marker_func=boundary_marker, component=0)

    def apply_bcs(R, K, U):
        return bc.apply(R, K, U, method="strong")
    
    # Solve
    print("--- Solving Poisson PDE ---")
    U_final = np.zeros(V.ndofs)
    U_final = solve_newton_raphson(U_final, assembler.assemble, apply_bcs)

    # Compute error
    def l2_error_form(u_gp, grad_u, x_gp, e):
        x, y = x_gp
        u_ex = u_exact(x, y)
        return (u_gp - u_ex)**2
    
    total_error_sq = assemble_scalar(V, integrand=l2_error_form, u_sol=U_final, quad_degree=2)
    L2_error = np.sqrt(total_error_sq)

    print(f"L2 Norm Error: {L2_error:.6e}")

    export_vtu(V, U_final, "results/poisson.vtu", field_name="Scalar Field")    
    export_vtu(V, U_final, "results/poisson_error.vtu", field_name="Error", n_vis_pts=4, exact_func=u_exact)