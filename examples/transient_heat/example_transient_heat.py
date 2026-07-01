import numpy as np
import time

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace
from fem_engine.bcs import DirichletBC
from fem_engine.assembler import Assembler, assemble_scalar
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.element import Quad4
from fem_engine.postprocess import export_vtu

# 1. Physics Parameters
L, W = 2.0, 1.0
k_diff = 1.0 # Thermal conductivity
rho_c = 1.0 # Volumetric heat capacity

# 2. The Weak Form (with theta-method) ---
# Regarding the 'state' argument at the end: The Assembler automatically 
# feeds 'u_old', 'grad_u_old', 'dt', and 'theta' into this
def transient_heat_weak_form(N, B_x, u, grad_u, x, e, state):
    dt = state['dt']
    theta = state['theta']
    u_old = state['u_old'] # Interpolated automatically
    grad_u_old = state['grad_u_old'] # Gradient interpolated automatically
    
    # 1. Rate of Change (Mass Matrix)
    # rho_c * (u - u_old) / dt
    transient_term = rho_c * np.outer(N, (u - u_old) / dt)
    
    # 2. Heat Flux (Stiffness Matrix)
    flux_now = k_diff * (B_x.T @ grad_u)
    flux_old = k_diff * (B_x.T @ grad_u_old)
    
    # Average the fluxes based on the chosen time-stepping scheme
    diffusion_term = theta * flux_now + (1.0 - theta) * flux_old
    
    # 3. Source term (assuming f = 0 for this example)
    # If we had a heat source, we would add: -np.outer(N, [source_val])
    
    return transient_term + diffusion_term

if __name__ == "__main__":
    # 3. Mesh and Space
    mesh = create_rectangle_mesh(L=L, W=W, n_x=20, n_y=10, x0=0.0, y0=0.0)
    V = FunctionSpace(mesh, Quad4(), n_components=1)
    
    # Initialize Assembler (it detects 'state' in the signature automatically)
    assembler = Assembler(V, transient_heat_weak_form, quad_degree=2)
    
    # 4. Boundary Conditions ---
    def left_wall_marker(x, y):
        return abs(x) < 1e-6
        
    def right_wall_marker(x, y):
        return abs(x - L) < 1e-6

    # Heat up the left wall to 100.0, keep the right wall frozen at 0.0
    bcs = [
        DirichletBC(V, value=100.0, boundary_marker_func=left_wall_marker, component=0),
        DirichletBC(V, value=0.0, boundary_marker_func=right_wall_marker, component=0)
    ]

    def apply_bcs(R, K, U):
        for bc in bcs:
            R, K = bc.apply(R, K, U, method="strong")
        return R, K

    # 5. Time Stepping Setup
    dt = 0.05
    t_end = 5.0
    current_time = 0.0
    
    # Theta = 0.5 is Crank-Nicolson (2nd order)
    # Theta = 1.0 is Backward Euler (1st order)
    theta = 0.5 
    
    # Initial Condition: U = 0 everywhere
    U_n = np.zeros(V.ndofs)
    
    print(f"--- Starting Transient Heat Solver (Theta = {theta}) ---")
    
    # 6. The Time Stepping and Solve Loop ---
    step = 0
    while current_time < t_end:
        current_time += dt
        step += 1
        print(f"\nStep {step} | Time: {current_time:.3f}s")
        
        # We package the history into global_params.
        # The Assembler handles slicing and interpolating this at the Gauss points.
        global_params = {
            'dt': dt,
            'theta': theta,
            'u_old': U_n  # Pass the global array
        }
        
        # Define a single-argument wrapper for the Newton-Raphson solver, otherwise for now it will crash
        def current_assembly(U_guess):
            return assembler.assemble(U_guess, global_params=global_params)
            
        # Solve the nonlinear step (or linear in this case, converges in 1 iteration)
        start_solve = time.time()
        U_next = solve_newton_raphson(U_n.copy(), current_assembly, apply_bcs)
        print(f"-> Solved in {time.time() - start_solve:.3f}s")
        
        # Update history for the next step
        U_n = U_next.copy()

        export_vtu(V, U_next, f"results/temperature_{step:04d}.vtu", field_name="Temperature")

    print("\nComplete")
    # 7. Steady-State L2 Error Check
    print("\n--- Verifying Steady-State Accuracy ---")
    
    def l2_steady_state_error(u_gp, grad_u, pos_gp, e):
        x, y = pos_gp
        u_exact = 100.0 * (1.0 - x / L)
        return (u_gp - u_exact)**2

    l2_error = np.sqrt(assemble_scalar(V, integrand=l2_steady_state_error, u_sol=U_n, quad_degree=2))
    
    print(f"Final L2 Error Norm: {l2_error:.4e}")