import numpy as np
import os
import time
import jax.numpy as jnp

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace, MixedFunctionSpace
from fem_engine.element import Quad4
from fem_engine.assembler_jax import MixedAssembler
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.postprocess import interpolate_to_nodes

# 1. PARAMETERS
L, W = 1.0, 1.0
n_x, n_y = 96, 96

dt = 5.0e-6
theta = 0.5
lmbda = 1.0e-2
M = 1.0

# 2. WEAK FORM
def cahn_hilliard_weak_form(mapped, pos_gp, e, state):
    """
    Coupled Mixed Formulation for Cahn-Hilliard.
    mapped[0] -> Concentration c
    mapped[1] -> Chemical potential mu
    """
    (N_c, B_c, c, grad_c) = mapped[0]
    (N_mu, B_mu, mu, grad_mu) = mapped[1]

    # Extract historical mixed state fields passed via global_params
    c_old, grad_c_old = state['U_old'][0]
    mu_old, grad_mu_old = state['U_old'][1]
    
    # Extract constants
    dt_state = state['dt']
    theta_state = state['theta']
    lmbda_state = state['lmbda']
    M_state = state['M']

    # Flatten arrays to cleanly avoid JAX/NumPy (N,) + (N, 1) broadcasting bugs
    c_val = c[0]
    mu_val = mu[0]
    c_old_val = c_old[0]
    
    grad_c = grad_c.flatten()
    grad_mu = grad_mu.flatten()
    grad_mu_old = grad_mu_old.flatten()

    # Theta-method (Crank-Nicolson) for the chemical potential gradient
    grad_mu_mid = (1.0 - theta_state) * grad_mu_old + theta_state * grad_mu
    
    # 1. First Equation: Concentration
    # R_c = \int (c - c_old)/dt * q dx + \int M * grad_mu_mid . grad_q dx
    transient_c = N_c * (c_val - c_old_val) / dt_state
    diffusion_c = M_state * jnp.dot(B_c.T, grad_mu_mid)
    R_c = transient_c + diffusion_c

    # 2. Second Equation: Chemical Potential
    # f(c) = 100 * c^2 * (1-c)^2
    # df/dc = 100 * (2c(1-c)^2 - 2c^2(1-c))
    dfdc = 100.0 * (2.0 * c_val * (1.0 - c_val)**2 - 2.0 * c_val**2 * (1.0 - c_val))
    
    # R_mu = \int \mu * v dx - \int df/dc * v dx - \int \lambda * grad_c . grad_v dx
    term1_mu = N_mu * mu_val
    term2_mu = -N_mu * dfdc
    term3_mu = -lmbda_state * jnp.dot(B_mu.T, grad_c)
    R_mu = term1_mu + term2_mu + term3_mu

    return [R_c, R_mu]


# 3. MAIN LOOP
if __name__ == "__main__":
    os.makedirs("results_cahn_hilliard", exist_ok=True)
    
    print("Creating structured quad mesh...")
    mesh = create_rectangle_mesh(L=L, W=W, n_x=n_x, n_y=n_y, x0=0.0, y0=0.0)
    
    # Function spaces
    V_c = FunctionSpace(mesh, Quad4(), n_components=1)
    V_mu = FunctionSpace(mesh, Quad4(), n_components=1)
    
    # Coupled Mixed Space
    V = MixedFunctionSpace([V_c, V_mu])

    # Initialize the JAX Assembler
    assembler = MixedAssembler(V, cahn_hilliard_weak_form, quad_degree=2)
    
    # Initialize arrays
    np.random.seed(42)
    U_n = np.zeros(V.ndofs)
    
    # Set the initial condition for concentration: c = 0.63 + small noise
    # Since V_c is the first space in the MixedFunctionSpace, its DOFs run from 0 to V_c.ndofs-1
    c_dofs = np.arange(V_c.ndofs)
    def initial_c(x, y):
        return 0.63 + 0.02 * (0.5 - np.random.rand())

    U_n[c_dofs] = interpolate_to_nodes(V_c, initial_c)
    
    def apply_bcs(R, K, U):
        # Our mixed Cahn-Hilliard relies solely on natural homogeneous Neumann boundary conditions (no-flux)
        # We simply bypass any modifications to R and K.
        return R, K

    # Time loop definition
    n_steps = 300
    t_end = n_steps * dt
    current_time = 0.0
    step = 0
    
    # Export initial state (Step 0)
    c_sol, mu_sol = V.split(U_n)
    from fem_engine.postprocess import CachedVTUExporter
    exporter = CachedVTUExporter(V_c, field_name="Concentration")
    exporter.export(c_sol, f"results_cahn_hilliard/cahn_hilliard_{step:04d}.vtu")

    print(f"\n--- Starting Cahn-Hilliard Mixed JAX Solver ({n_x}x{n_y} grid) ---")
    
    while current_time < t_end:
        step += 1
        current_time += dt
        
        print(f"\nStep {step}/{n_steps} | Time: {current_time:.2e}s")
        
        # Package the historical mixed state for the JAX assembler
        # 'U_old' with size matching V.ndofs automatically maps directly to all spaces
        global_params = {
            'U_old': U_n,
            'dt': dt,
            'theta': theta,
            'lmbda': lmbda,
            'M': M
        }
        
        # Lock in historical inputs for the NR routine callback
        def current_assembly(U_guess):
            return assembler.assemble(U_guess, global_params=global_params)
            
        start_solve = time.time()
        
        # Non-linear Newton solve for the Mixed Formulation
        U_next = solve_newton_raphson(U_n.copy(), current_assembly, apply_bcs, nr_tol=1e-6, max_iter=10)
        
        print(f"-> Newton converged in {time.time() - start_solve:.3f}s")
        
        # Update history
        U_n = U_next.copy()
        
        # Export frames
        c_sol, mu_sol = V.split(U_n)
        start_export = time.time()
        exporter.export(c_sol, f"results_cahn_hilliard/cahn_hilliard_{step:04d}.vtu")
        print(f"-> VTU Export took {time.time() - start_export:.4f}s")
        
    print("\nPhase-separation simulation complete.")