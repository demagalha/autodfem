import numpy as np
from scipy.sparse.linalg import spsolve

def solve_newton_raphson(U_initial, compute_system_callback, apply_bcs_callback, nr_tol=1e-8, max_iter=10):
    U = U_initial.copy()

    print("Starting Newton-Raphson Solver...")
    for it in range(max_iter):
        
        # 1. Evaluate Residual and Stiffness
        R_global, K_global = compute_system_callback(U)
        
        # 2. Apply Boundary Conditions
        R_global, K_global = apply_bcs_callback(R_global, K_global, U)
        
        # 3. Check Convergence
        res_norm = np.linalg.norm(R_global)
        print(f"  Iteration {it}: Residual Norm = {res_norm:.3e}")
        if res_norm < nr_tol:
            print("  Converged successfully!\n")
            break
            
        # 4. Solve Linear System
        K_global = K_global.tocsr()
        delta_U = spsolve(K_global, -R_global)
        U += delta_U

    return U

def solve_linear(K, F):
    """Directly solves the linear system K * U = F using a sparse solver."""
    # spsolve requires the matrix to be in CSR or CSC format
    return spsolve(K.tocsr(), F)