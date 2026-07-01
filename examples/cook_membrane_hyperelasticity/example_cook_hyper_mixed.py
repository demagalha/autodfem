import numpy as np
import gmsh
import os
import time
import matplotlib.pyplot as plt

from fem_engine.external_mesh import import_mesh
from fem_engine.mesh import FunctionSpace, MixedFunctionSpace
from fem_engine.bcs import DirichletBC, NeumannBC
from fem_engine.assembler import MixedAssembler
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.element import Quad9, Quad0
from fem_engine.postprocess import export_vtu, extrapolate_mixed_gauss_to_nodes, evaluate_field_at_point

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

# 2. Mixed Q2P0 Hyperelastic Weak Form
def mixed_spp1748_hyperelasticity(mapped, pos_gp, e):
    """
    Mixed u-p formulation to prevent volumetric locking.
    mapped[0] -> Displacement space V_u (Quad9)
    mapped[1] -> Pressure space V_p (Quad0)
    """
    # 1. Unpack the mapped fields
    N_u, B_u, u_gp, grad_u = mapped[0]
    N_p, B_p, p_gp, grad_p = mapped[1]
    
    # 2. Kinematics
    F = np.eye(2, dtype=grad_u.dtype) + grad_u.T
    J = F[0,0] * F[1,1] - F[0,1] * F[1,0]
    
    FinvT = np.zeros((2, 2), dtype=grad_u.dtype)
    FinvT[0,0] =  F[1,1] / J
    FinvT[0,1] = -F[1,0] / J
    FinvT[1,0] = -F[0,1] / J
    FinvT[1,1] =  F[0,0] / J
    
    # Extract the scalar pressure from the P0 space
    p = p_gp[0]
    
    # 3. First Piola-Kirchhoff Stress (Using independent 'p')
    # P = mu * F + (p - mu) * F^-T
    coeff = p - mu
    P = np.zeros((2, 2), dtype=grad_u.dtype)
    P[0,0] = mu * F[0,0] + coeff * FinvT[0,0]
    P[0,1] = mu * F[0,1] + coeff * FinvT[0,1]
    P[1,0] = mu * F[1,0] + coeff * FinvT[1,0]
    P[1,1] = mu * F[1,1] + coeff * FinvT[1,1]
    
    # 4. Residual for Displacement (Equilibrium)
    R_u = B_u.T @ P.T  # Shape: (9 nodes, 2 components)
    
    # 5. Residual for Pressure (Volumetric Constraint)
    # Scaled by 1/lambda_ to keep the tangent matrix well-conditioned
    # Equation: p/lambda - 0.5*(J^2 - 1) = 0
    vol_eq = (p / lambda_) - 0.5 * (J**2 - 1.0)
    R_p = N_p * vol_eq  # Shape: (1 node, 1 component)
    
    return [R_u, R_p]

def compute_spp1748_stress(u_gp, grad_u):
    """Classic pure-displacement recovery for plotting purely from u_sol"""
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
    return sigma[0,0]

def compute_true_spp1748_mixed_stress(u_gp, grad_u, p_gp, grad_p):
    """
    True mixed Cauchy stress recovery.
    Uses the independent pressure field solved by the system.
    """
    F = np.eye(2) + grad_u.T
    J = F[0,0]*F[1,1] - F[0,1]*F[1,0]
    
    # Extract the scalar pressure from the P0 space evaluation
    p = p_gp[0]
    
    # Compute the Left Cauchy-Green tensor: b = F * F^T
    b = F @ F.T 
    
    # Compute true Cauchy stress: sigma = (1/J) * (mu * b + (p - mu) * I)
    sigma = np.zeros((2,2))
    sigma[0,0] = (mu * b[0,0] + (p - mu)) / J
    sigma[0,1] = (mu * b[0,1]) / J
    sigma[1,0] = (mu * b[1,0]) / J
    sigma[1,1] = (mu * b[1,1] + (p - mu)) / J
    
    # Return Sigma_xx for visualization
    return sigma[0,0]

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

    for line in [l1, l2, l3, l4]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, n_elements + 1)
    
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    gmsh.model.geo.mesh.setRecombine(2, surf)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.model.mesh.setOrder(2)
        
    gmsh.write(filename)
    gmsh.finalize()

# 4. Main Execution & Convergence Loop
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    element_densities = [2, 4, 8, 12, 16, 32]
    
    dofs_list = []
    uy_tip_list = []
    
    print("MIXED Q2P0 COOK'S MEMBRANE CONVERGENCE STUDY\n")

    total_start_time = time.time()
    
    for n_el in element_densities:
        print(f"\n--- Solving for {n_el}x{n_el} element grid ---")
        msh_file = f"results/cooks_membrane_mixed_{n_el}x{n_el}.msh"
        
        generate_cooks_membrane(msh_file, n_el)
        mesh = import_mesh(msh_file, element_type="quad9")
        
        # Create Mixed Function Space
        V_u = FunctionSpace(mesh, Quad9(), n_components=2)
        V_p = FunctionSpace(mesh, Quad0(), n_components=1)
        V_mixed = MixedFunctionSpace([V_u, V_p])
        
        assembler = MixedAssembler(V_mixed, mixed_spp1748_hyperelasticity, quad_degree=5)

        U_current = np.zeros(V_mixed.ndofs)
        n_steps = 4 

        # Incremental Load
        for step in range(1, n_steps + 1):
            current_traction = p0 * (step / n_steps) 
            
            def apply_bcs(R, K, U):
                tol = 1e-5
                
                # BCs ONLY apply to the displacement space (V_u), so we use offset=0
                bc_neumann = NeumannBC(V_u, load_vector=[0.0, current_traction], boundary_marker_func=lambda x, y: x > x_right - tol)
                R, K = bc_neumann.apply(R, K, U, offset=0)
                
                bcs_dirichlet = [
                    DirichletBC(V_u, value=0.0, boundary_marker_func=lambda x, y: x < tol, component=0),
                    DirichletBC(V_u, value=0.0, boundary_marker_func=lambda x, y: x < tol, component=1)
                ]
                
                for bc in bcs_dirichlet:
                    R, K = bc.apply(R, K, U, offset=0, method="strong", is_linear=False)
                    
                return R, K

            # Solve the non-linear mixed system
            U_current = solve_newton_raphson(U_current, assembler.assemble, apply_bcs, nr_tol=1e-5, max_iter=5)

        # Split the mixed solution back into displacement and pressure fields
        u_sol, p_sol = V_mixed.split(U_current)

        # Evaluate tip displacement strictly on V_u
        target_pt = np.array([x_right - 1e-5, y_top_right - 1e-5])
        val = evaluate_field_at_point(mesh, V_u, u_sol, target_pt)
        uy_tip = val[1]
        
        dofs_list.append(V_mixed.ndofs)
        uy_tip_list.append(uy_tip)
        
        print(f"-> Mixed DOFs: {V_mixed.ndofs} | Final Tip Uy: {uy_tip:.5f}")

        if n_el == element_densities[-1]:
            # Export Displacement (from V_u) and independent Pressure (from V_p)
            export_vtu(V_u, u_sol, "results/cooks_mixed_disp.vtu", field_name="Displacement")
            export_vtu(V_p, p_sol, "results/cooks_mixed_pressure.vtu", field_name="Pressure")

            # Calculate and export stress
            V_stress, U_stress = extrapolate_mixed_gauss_to_nodes(V_u, u_sol, V_p, p_sol, compute_true_spp1748_mixed_stress, n_components=1)
            export_vtu(V_stress, U_stress, "results/cooks_mixed_sigmaxx.vtu", field_name="Sigma_xx")

    print(f"\nTotal study completed in {time.time() - total_start_time:.2f} s")

    # 5. Convergence Plotting
    print("\nGenerating Convergence Plot...")
    
    dofs_array = np.array(dofs_list)
    uy_array = np.array(uy_tip_list)

    plt.figure(figsize=(9, 6))
    
    plt.plot(dofs_array, uy_array, marker='o', linestyle='-', color='g', linewidth=2, label='FEM Mixed (Q2P0) Uy')
    
    plt.title("Cook's Membrane: Q2P0 Mixed Displacement Convergence", fontsize=14, fontweight='bold')
    plt.xlabel('Total Degrees of Freedom (DOFs)', fontsize=12)
    plt.ylabel(r'Vertical Displacement at Top-Right, $u_y$', fontsize=12)
    plt.xscale('log')
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig('results/cooks_membrane_mixed_convergence.png')
    print("Plot saved to 'results/cooks_membrane_mixed_convergence.png'.")
    plt.show()