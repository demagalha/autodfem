import numpy as np

from fem_engine.mesh import create_quadratic_rectangle_mesh, create_rectangle_mesh, FunctionSpace, MixedFunctionSpace
from fem_engine.element import Quad9, Quad4
from fem_engine.assembler import MixedAssembler
from fem_engine.bcs import DirichletBC
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.postprocess import export_vtu, evaluate_field_at_point


# 1. Mesh
L, W = 1.0, 1.0
elements_x, elements_y = 20, 20  # Increased resolution

mesh = create_quadratic_rectangle_mesh(L, W, n_x=elements_x, n_y=elements_y, x0=0.0, y0=0.0)

# We could just use this just as well, since mesh !=! element space...
# The number of dofs in the assembled system is the same when, the mesh "only" helps with the geometric mapping
#mesh = create_rectangle_mesh(L, W, n_x=elements_x, n_y=elements_y, x0=0.0, y0=0.0)


# 2. Spaces (Q2/Q1 Taylor-Hood)
V_u = FunctionSpace(mesh, Quad9(), n_components=2)  
V_p = FunctionSpace(mesh, Quad4(), n_components=1)  

# For a Mixed Space this is the usual way we'll pass things
V = MixedFunctionSpace([V_u, V_p])

# 3. Weak form
# Instead of other examples, whe working with a mixed form we have multiple unknowns
# If we'd have a third one we would use mapped[2]
# The idea, though, is the same for earlier: we have access, in order, to the Basis Fun and first derivatives (if it makes sense), the field and its derivative 
def navier_stokes_weak_form(mapped, gp, e):
    (N_u, B_u, u, grad_u) = mapped[0]
    (N_p, B_p, p, grad_p) = mapped[1]

    # Because U=1.0 and L=1.0, Re = 1/nu. 
    # Let's set Re = 100 for the benchmark
    Re = 100.0
    nu = 1.0 / Re

    # 1. Viscous term
    visc = nu * (B_u.T @ grad_u)   # (9,2)

    # 2. Pressure gradient
    # the [0] is mostly due to issues with numpy broadcasting and the Dual class... something to fix in the future
    p_val = p[0]
    
    p_div_v = np.zeros_like(visc)
    for i in range(len(N_u)):
        p_div_v[i, 0] = B_u[0, i] * p_val
        p_div_v[i, 1] = B_u[1, i] * p_val
    
    # Can be just: p_div_v = B_u.T * p_val

    # 3. Non-linear Advection term: (u . nabla) u
    # 'u' is (2,), 'grad_u' is (2,2). 
    # 'u @ grad_u' gives the convective acceleration vector (2,)
    # np.outer distributes it to the 9 nodes -> (9,2)
    advection = np.outer(N_u, u @ grad_u)

    # Momentum residual
    R_u = visc - p_div_v + advection

    # 4. Continuity equation
    div_u = grad_u[0, 0] + grad_u[1, 1]
    
    R_p = -np.outer(N_p, [div_u])

    return [R_u, R_p]

assembler = MixedAssembler(V, navier_stokes_weak_form, quad_degree=3)

# 4. Boundary conditions
def left_wall(x, y): return abs(x) < 1e-6
def right_wall(x, y): return abs(x - L) < 1e-6
def top_wall(x, y): return abs(y - W) < 1e-6
def bottom_wall(x, y): return abs(y) < 1e-6

def apply_bcs(R, K, U):
    offset_u = 0
    offset_p = V_u.ndofs

    # Lid-driven cavity:
    # top: u = (1,0)
    bc_top_x = DirichletBC(V_u, value=1.0, boundary_marker_func=top_wall, component=0)
    bc_top_y = DirichletBC(V_u, value=0.0, boundary_marker_func=top_wall, component=1)

    # no-slip elsewhere
    bc_zero_x = DirichletBC(V_u, value=0.0, boundary_marker_func=lambda x,y: left_wall(x,y) or right_wall(x,y) or bottom_wall(x,y), component=0)
    bc_zero_y = DirichletBC(V_u, value=0.0, boundary_marker_func=lambda x,y: left_wall(x,y) or right_wall(x,y) or bottom_wall(x,y), component=1)

    for bc in [bc_top_x, bc_top_y, bc_zero_x, bc_zero_y]:
        R, K = bc.apply(R, K, U, offset=offset_u)

    # pressure fix (otherwise it is defined only up still a constant)
    bc_p = DirichletBC(V_p, value=0.0, boundary_marker_func=lambda x,y: abs(x)<1e-6 and abs(y)<1e-6)
    R, K = bc_p.apply(R, K, U, offset=offset_p)

    return R, K


# 5. Solve
U0 = np.zeros(V.ndofs)

U_sol = solve_newton_raphson(U0, assembler.assemble, apply_bcs)

print("Solved Nav Stokes (Steady State)")

import matplotlib.pyplot as plt

# Post Processing

# Split solution 
U_u, U_p = V.split(U_sol)

export_vtu(V_u, U_u, "results/velocity_cavity.vtu", field_name="Velocity", n_vis_pts=5)
export_vtu(V_p, U_p, "results/pressure_cavity.vtu", field_name="Pressure")

ghia_y = np.array([
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000
])

u_sim = []

for y in ghia_y:
    val = evaluate_field_at_point(mesh, V_u, U_u, np.array([0.5, y]))
    u_sim.append(val[0])   # x-velocity

print("Top point:", u_sim[0], "should be ~1.0")


ghia_u_100 = np.array([
    1.00000,
    0.84123,
    0.78871,
    0.73722,
    0.68717,
    0.23151,
    0.00332,
    -0.13641,
    -0.20581,
    -0.21090,
    -0.15662,
    -0.10150,
    -0.06434,
    -0.04775,
    -0.04192,
    -0.03717,
    0.00000
])

y_fine = np.linspace(0, 1, 200)
u_fine = []

for y in y_fine:
    val = evaluate_field_at_point(mesh, V_u, U_u, np.array([0.5, y]))
    u_fine.append(val[0])

u_fine = np.array(u_fine)

plt.plot(ghia_u_100, ghia_y, 'ro', label="Ghia Re=100")
#plt.plot(u_sim, ghia_y, 'b--', label="Solver")
plt.plot(u_fine, y_fine, 'b-', label="Solver")

plt.xlabel("u velocity")
plt.ylabel("y")
plt.legend()
plt.gca()
plt.title("Lid-driven cavity comparison (Re=100)")
plt.show()


u_sim = np.array(u_sim)
ghia = ghia_u_100

err = u_sim - ghia

eps = 1e-12
percent_error = 100.0 * err / (np.abs(ghia) + eps)

print(percent_error)

l2_rel = np.linalg.norm(u_sim - ghia) / np.linalg.norm(ghia)
print("L2 relative error:", 100 * l2_rel, "%")


plt.figure()
plt.plot(percent_error, ghia_y, 'k-o')
plt.gca()
plt.xlabel("Percent error (%)")
plt.ylabel("y")
plt.title("Ghia comparison error (Re=100)")
plt.show()

