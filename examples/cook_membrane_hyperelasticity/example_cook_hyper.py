import numpy as np
import gmsh
import os
import time
import matplotlib.pyplot as plt

from fem_engine.external_mesh import import_mesh
from fem_engine.mesh import FunctionSpace
from fem_engine.bcs import DirichletBC, NeumannBC
from fem_engine.assembler import Assembler
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.element import Quad9
from fem_engine.postprocess import export_vtu, extrapolate_gauss_to_nodes, evaluate_field_at_point

# 1. Physics Parameters (SPP-1748 Benchmark Table 2.1)
lambda_ = 432.099
mu = 185.185
p0 = 20.0 # Constant shear load

# Cook's Membrane Geometry
x_left = 0.0
x_right = 48.0
y_bottom_left = 0.0
y_top_left = 44.0
y_bottom_right = 44.0
y_top_right = 60.0

# 2. SPP-1748 Hyperelastic Weak Form (psi_1)
def spp1748_hyperelasticity(N, B_x, u_gp, grad_u, x_gp, e):
    """
    1st PK Stress derived from: 
    S = 0.5 * lambda * (J^2 - 1) * C^-1 + mu * (I - C^-1)
    P = F * S = mu * F + [0.5 * lambda * (J^2 - 1) - mu] * F^-T
    """
    F = np.eye(2, dtype=grad_u.dtype) + grad_u.T
    J = F[0,0] * F[1,1] - F[0,1] * F[1,0]
    
    # F^-T (Inverse Transpose)
    FinvT = np.zeros((2, 2), dtype=grad_u.dtype)
    FinvT[0,0] =  F[1,1] / J
    FinvT[0,1] = -F[1,0] / J
    FinvT[1,0] = -F[0,1] / J
    FinvT[1,1] =  F[0,0] / J
    
    coeff = 0.5 * lambda_ * (J**2 - 1.0) - mu
    
    P = np.zeros((2, 2), dtype=grad_u.dtype)
    P[0,0] = mu * F[0,0] + coeff * FinvT[0,0]
    P[0,1] = mu * F[0,1] + coeff * FinvT[0,1]
    P[1,0] = mu * F[1,0] + coeff * FinvT[1,0]
    P[1,1] = mu * F[1,1] + coeff * FinvT[1,1]
    
    return B_x.T @ P.T

def compute_spp1748_stress(u_gp, grad_u):
    """Cauchy stress recovery for post-processing: sigma = (P * F^T) / J"""
    F = np.eye(2) + grad_u.T
    J = F[0,0]*F[1,1] - F[0,1]*F[1,0]

    FinvT = np.zeros((2,2))
    FinvT[0,0] =  F[1,1]/J
    FinvT[0,1] = -F[1,0]/J
    FinvT[1,0] = -F[0,1]/J
    FinvT[1,1] =  F[0,0]/J

    coeff = 0.5 * lambda_ * (J**2 - 1.0) - mu
    P = np.zeros((2,2))
    P[0,0] = mu * F[0,0] + coeff * FinvT[0,0]
    P[0,1] = mu * F[0,1] + coeff * FinvT[0,1]
    P[1,0] = mu * F[1,0] + coeff * FinvT[1,0]
    P[1,1] = mu * F[1,1] + coeff * FinvT[1,1]

    sigma = (P @ F.T) / J
    sigma_xx = sigma[0,0]
    
    return sigma_xx

# 3. Mesh Generation
def generate_cooks_membrane(filename, n_elements):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CooksMembrane")

    p1 = gmsh.model.geo.addPoint(x_left, y_bottom_left, 0)
    p2 = gmsh.model.geo.addPoint(x_right, y_bottom_right, 0)
    p3 = gmsh.model.geo.addPoint(x_right, y_top_right, 0)
    p4 = gmsh.model.geo.addPoint(x_left, y_top_left, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    # Force a structured grid
    for line in [l1, l2, l3, l4]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, n_elements + 1)
    
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    gmsh.model.geo.mesh.setRecombine(2, surf)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Force strict 9-node Quad9 elements
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.model.mesh.setOrder(2)

    gmsh.write(filename)
    gmsh.finalize()

# 4. Main & Convergence Loop
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Define the mesh densities to test (number of elements per edge)
    element_densities = [2, 4, 8, 12, 16, 20]
    
    dofs_list = []
    uy_tip_list = []
    
    print("\n" + "="*60)
    print("STARTING SPP-1748 COOK'S MEMBRANE CONVERGENCE STUDY")
    print("="*60)

    total_start_time = time.time()
    
    for n_el in element_densities:
        print(f"\n--- Solving for {n_el}x{n_el} element grid ---")
        msh_file = f"results/cooks_membrane_{n_el}x{n_el}.msh"
        
        generate_cooks_membrane(msh_file, n_el)
        mesh = import_mesh(msh_file, element_type="quad9")
        
        V = FunctionSpace(mesh, Quad9(), n_components=2)
        assembler = Assembler(V, spp1748_hyperelasticity, quad_degree=3)

        U_current = np.zeros(V.ndofs)
        n_steps = 4 

        # Incremental Load
        for step in range(1, n_steps + 1):
            current_traction = p0 * (step / n_steps) 
            
            def apply_bcs(R, K, U):
                tol = 1e-5
                
                # Apply Upward Traction to the right edge (Neumann)
                bc_neumann = NeumannBC(V, load_vector=[0.0, current_traction], boundary_marker_func=lambda x, y: x > x_right - tol)
                R, K = bc_neumann.apply(R, K, U)
                
                # Fully Clamp the left edge (Dirichlet)
                def left_edge(x, y): return x < tol
                bcs_dirichlet = [
                    DirichletBC(V, value=0.0, boundary_marker_func=left_edge, component=0),
                    DirichletBC(V, value=0.0, boundary_marker_func=left_edge, component=1)
                ]
                
                for bc in bcs_dirichlet:
                    R, K = bc.apply(R, K, U)
                return R, K

            # Solve
            U_current = solve_newton_raphson(U_current, assembler.assemble, apply_bcs, nr_tol=1e-5, max_iter=5)

        # Evaluate displacement at top-right tip
        target_pt = np.array([x_right, y_top_right])
        val = evaluate_field_at_point(mesh, V, U_current, target_pt)
        uy_tip = val[1]
        
        dofs_list.append(V.ndofs)
        uy_tip_list.append(uy_tip)
        
        print(f"-> DOFs: {V.ndofs} | Final Tip Uy: {uy_tip:.5f}")

        # Export the finest mesh results for ParaView
        if n_el == element_densities[-1]:
            export_vtu(V, U_current, "results/cooks_spp1748_disp.vtu", field_name="Displacement")
            V_stress, U_stress = extrapolate_gauss_to_nodes(V, U_current, compute_spp1748_stress, n_components=1)
            export_vtu(V_stress, U_stress, "results/cooks_spp1748_stress.vtu", field_name="Sigma_xx")

    print(f"\nTotal study completed in {time.time() - total_start_time:.2f} s")

    # 5. Convergence Plotting
    print("\nGenerating Convergence Plot...")
    
    dofs_array = np.array(dofs_list)
    uy_array = np.array(uy_tip_list)

    plt.figure(figsize=(9, 6))
    
    # Plot FEM convergence curve
    plt.plot(dofs_array, uy_array, marker='o', linestyle='-', color='g', linewidth=2, label='FEM (Quad9) Uy')
    
    plt.title("Cook's Membrane: Vertical Tip Displacement Convergence", fontsize=14, fontweight='bold')
    plt.xlabel('Degrees of Freedom (DOFs)', fontsize=12)
    plt.ylabel(r'Vertical Displacement at Top-Right, $u_y$', fontsize=12)
    plt.xscale('log')
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig('results/cooks_membrane_convergence.png')
    print("Plot saved to 'results/cooks_membrane_convergence.png'.")
    plt.show()