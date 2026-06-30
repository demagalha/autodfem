import numpy as np
import gmsh
import os
import time

from fem_engine.external_mesh import import_mesh
from fem_engine.mesh import FunctionSpace
from fem_engine.bcs import DirichletBC, NeumannBC
from fem_engine.assembler import Assembler
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.element import Quad4
from fem_engine.postprocess import export_vtu, average_at_nodes, evaluate_field_at_point

# Physics & Geometry Parameters
E = 1e5 
nu = 0.3
lambda_ = E * nu / (1 - nu**2)
mu = E / (2 * (1 + nu))

L = 4.0 
W = 4.0
a = 1.0
T_inf = 10.0 # Far-field load

# Analytical Kirsch Solutions (for Boundary Tractions)
def kirsch_stress_tensor(x, y, T, a):
    """Returns the exact analytical 2D Cauchy stress tensor at (x, y)."""
    r = np.sqrt(x**2 + y**2) + 1e-15
    theta = np.arctan2(y, x)
    
    sig_r = (T / 2) * (1 - a**2 / r**2) + (T / 2) * (1 - 4*a**2 / r**2 + 3*a**4 / r**4) * np.cos(2*theta)
    sig_t = (T / 2) * (1 + a**2 / r**2) - (T / 2) * (1 + 3*a**4 / r**4) * np.cos(2*theta)
    tau_rt = -(T / 2) * (1 + 2*a**2 / r**2 - 3*a**4 / r**4) * np.sin(2*theta)
    
    c, s = np.cos(theta), np.sin(theta)
    sig_x = sig_r * c**2 + sig_t * s**2 - 2 * tau_rt * s * c
    sig_y = sig_r * s**2 + sig_t * c**2 + 2 * tau_rt * s * c
    tau_xy = (sig_r - sig_t) * s * c + tau_rt * (c**2 - s**2)
    
    return np.array([
        [sig_x, tau_xy], 
        [tau_xy, sig_y]
    ])

# Traction functions for the boundaries
def traction_right_edge(x, y):
    """Traction on x=L boundary. Normal vector n = [1, 0]. T = Sigma * n"""
    sigma = kirsch_stress_tensor(x, y, T_inf, a)
    return [sigma[0, 0], sigma[1, 0]]  # [sig_xx, tau_xy]

def traction_top_edge(x, y):
    """Traction on y=W boundary. Normal vector n = [0, 1]. T = Sigma * n"""
    sigma = kirsch_stress_tensor(x, y, T_inf, a)
    return [sigma[0, 1], sigma[1, 1]]  # [tau_xy, sig_yy]

# Mesh Generation
def generate_quarter_plate_hole(filename, lc_far=0.4, lc_hole=0.015):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("PlateHoleExact")

    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, W)
    disk = gmsh.model.occ.addDisk(0, 0, 0, a, a)
    out, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("MathEval", 1)
    gmsh.model.mesh.field.setString(1, "F", f"{lc_hole} + ({lc_far} - {lc_hole}) * (sqrt(x*x + y*y) - {a}) / {L * 1.414 - a}")
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    
    for dim, tag in out:
        if dim == 2:
            gmsh.model.mesh.setRecombine(2, tag)
            
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()

# Weak Form
def linear_elasticity(N, B_x, u_gp, grad_u, x_gp, e):
    eps = 0.5 * (grad_u + grad_u.T)
    tr_eps = eps[0,0] + eps[1,1]
    
    sigma = np.zeros((2, 2), dtype=grad_u.dtype)
    sigma[0,0] = lambda_ * tr_eps + 2 * mu * eps[0,0]
    sigma[1,1] = lambda_ * tr_eps + 2 * mu * eps[1,1]
    sigma[0,1] = 2 * mu * eps[0,1]
    sigma[1,0] = 2 * mu * eps[1,0]
    
    return B_x.T @ sigma

def compute_sigma_xx(u_gp, grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    tr_eps = eps[0,0] + eps[1,1]
    return lambda_ * tr_eps + 2 * mu * eps[0,0]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    msh_file = "results/plate_hole_exact.msh"
    
    print("\n--- Generating Exact Traction Mesh (L=4.0) ---")
    generate_quarter_plate_hole(msh_file)
    mesh = import_mesh(msh_file, element_type="quad")
    
    V = FunctionSpace(mesh, Quad4(), n_components=2)
    assembler_el = Assembler(V, linear_elasticity, quad_degree=2)

    def apply_bcs(R, K, U):
        tol = 1e-5
        
        # Neumann BCs
        bcs_neumann = [
            NeumannBC(V, load_vector=traction_right_edge, boundary_marker_func=lambda x, y: x > L - tol),
            NeumannBC(V, load_vector=traction_top_edge, boundary_marker_func=lambda x, y: y > W - tol)
        ]
        
        for bc in bcs_neumann:
            R, K = bc.apply(R, K, U)
            
        # Symmetry Dirichlet BCs
        bcs_dirichlet = [
            DirichletBC(V, value=0.0, boundary_marker_func=lambda x, y: x < tol, component=0),
            DirichletBC(V, value=0.0, boundary_marker_func=lambda x, y: y < tol, component=1)
        ]
        
        for bc in bcs_dirichlet: 
            R, K = bc.apply(R, K, U)
            
        return R, K

    print("\n--- Solving Finite Plate with Exact Boundary Tractions ---")
    start_time = time.time()
    U_final = solve_newton_raphson(np.zeros(V.ndofs), assembler_el.assemble, apply_bcs)
    print(f"Solve completed in {time.time() - start_time:.3f} seconds")

    V_stress_node, U_stress_node = average_at_nodes(V, U_final, compute_sigma_xx, n_components=1)
    target_pt = np.array([0.0, a])
    sigma_xx_tip = evaluate_field_at_point(mesh, V_stress_node, U_stress_node, target_pt)
    
    exact_tensor_tip = kirsch_stress_tensor(0.0, a, T_inf, a)
    sigma_xx_analytical_tip = exact_tensor_tip[0, 0]

    print(f"Applied Far-Field Kirsch Tensor : {T_inf:.2f}")
    print(f"Peak Stress at Tip : {sigma_xx_tip[0]:.2f}")
    print(f"Analytical sigma_xx at tip : {sigma_xx_analytical_tip:.6f}")

    export_vtu(V, U_final, "results/plate_hole_disp.vtu", field_name="Displacement")
    export_vtu(V_stress_node, U_stress_node, "results/plate_hole_sigmaxx.vtu", field_name="Sigma_xx")

    # Plotting Stress along the vertical symmetry line (x = 0)
    import matplotlib.pyplot as plt

    print("Extracting data along the vertical symmetry line (x=0)")
    
    # 50 sample points from the hole edge (y=a) to the top boundary (y=W)
    n_samples = 50
    y_line = np.linspace(a, W, n_samples)
    
    sig_xx_fem = np.zeros(n_samples)
    sig_xx_exact = np.zeros(n_samples)

    for i, y_val in enumerate(y_line):
        target_pt = np.array([0.0, y_val])
        
        # 1. Evaluate Numerical Stress
        fem_val = evaluate_field_at_point(mesh, V_stress_node, U_stress_node, target_pt)
        
        # Handle whether the framework returns a float or a 1D array, fix this later...
        sig_xx_fem[i] = fem_val[0] if isinstance(fem_val, (list, np.ndarray, tuple)) else fem_val
        
        # 2. Evaluate Analytical Stress
        exact_tensor = kirsch_stress_tensor(0.0, y_val, T_inf, a)
        sig_xx_exact[i] = exact_tensor[0, 0]

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot exact solution
    plt.plot(y_line, sig_xx_exact, 'k-', linewidth=2.0, label='Exact Analytical (Kirsch)')
    
    # Plot FEM solution as discrete points
    plt.plot(y_line, sig_xx_fem, 'ro', markersize=5, label='FEM (Quad4)')
    
    plt.title(r'Stress along Symmetry Line ($x=0$)', fontsize=14)
    plt.xlabel('Distance from center, y', fontsize=12)
    plt.ylabel(r'Horizontal Stress, $\sigma_{xx}$', fontsize=12)
    
    # Vertical dashed line for the hole boundary
    plt.axvline(x=a, color='gray', linestyle='--', label='Hole Edge (y=a)')
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('results/stress_decay_plot.png', dpi=300)
    print("Plot saved to 'results/stress_plot.png'")
    plt.show()