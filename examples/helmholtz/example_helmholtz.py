import numpy as np

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace, MixedFunctionSpace
from fem_engine.element import Quad4
from fem_engine.assembler import MixedAssembler
from fem_engine.fem_solver import solve_newton_raphson
from fem_engine.postprocess import export_vtu, evaluate_field_at_point

# Physical parameters
L = 1.0
W = 1.0

c = 343.0
rho0 = 1.225

frequency = 500.0 # Hz
omega = 2.0 * np.pi * frequency
k = omega / c

# Gaussian monopole Source
def q_source(x, y):
    sigma = 0.03 # Gaussian width
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    amplitude = 1.0 / (2.0 * np.pi * sigma**2)

    return amplitude * np.exp(-r2 / (2.0 * sigma**2))

# Mixed Helmholtz weak form
# A way to go around the restriction on AD through Dual (that currently does not support complex numbers)

def helmholtz_weak(mapped, pos_gp, e):

    (N_r, B_r, p_r, grad_pr) = mapped[0]
    (N_i, B_i, p_i, grad_pi) = mapped[1]

    x, y = pos_gp

    q = 0.0005 * q_source(x, y)

    # Real part
    # lap(pr) + k^2 pr = 0

    stiffness_r = B_r.T @ grad_pr
    mass_r  = -np.outer(N_r, k**2 * p_r)

    R_r = stiffness_r + mass_r

    # Imaginary part
    # lap(pi) + k^2 pi = -omega * rho * Q

    stiffness_i = B_i.T @ grad_pi
    mass_i  = -np.outer(N_i, k**2 * p_i)

    source = -np.outer(N_i, omega * rho0 * q)

    R_i = stiffness_i + mass_i + source

    return [R_r, R_i]

if __name__ == "__main__":

    mesh = create_rectangle_mesh(L=L,W=W,n_x=60,n_y=60,x0=0.0,y0=0.0)

    V_r = FunctionSpace(mesh, Quad4(), n_components=1)
    V_i = FunctionSpace(mesh, Quad4(), n_components=1)

    V = MixedFunctionSpace([V_r, V_i])

    assembler = MixedAssembler(V, helmholtz_weak, quad_degree=2)

    # Rigid walls:
    # del p/del n = 0
    # This is the natural BC of the weak form,
    # therefore NO boundary conditions are applied.

    def apply_bcs(R, K, U):
        return R, K

    U0 = np.zeros(V.ndofs)

    U = solve_newton_raphson(U0, assembler.assemble, apply_bcs)

    p_real, p_imag = V.split(U)

    # Microphone location

    mic = np.array([0.2,0.2])

    # Microphone evaluation

    Pr = evaluate_field_at_point(mesh, V_r, p_real, mic)
    Pi = evaluate_field_at_point(mesh, V_i, p_imag, mic)

    # remove possible array wrapping

    Pr = Pr[0] if np.ndim(Pr)>0 else Pr
    Pi = Pi[0] if np.ndim(Pi)>0 else Pi

    # complex magnitude

    Pmag = np.sqrt(Pr**2 + Pi**2)

    print("SPL at mic location is")
    pref = 2e-5
    spl = 20*np.log10(Pmag/pref)
    print(spl)

    export_vtu(V_r, p_real, "results/helmholtz_real.vtu", field_name="PressureReal")

    export_vtu(V_i, p_imag, "results/helmholtz_imag.vtu", field_name="PressureImag")

    # Export harmonic animation for ParaView

    n_frames = 100

    # Extract the real and imaginary arrays
    pr = p_real
    pi = p_imag

    for i in range(n_frames):

        # one period over the animation
        theta = 2.0 * np.pi * i / n_frames

        # p(t) = Re(P e^{jwt})
        pressure_t = pr * np.cos(theta) - pi * np.sin(theta)

        export_vtu(V_r, pressure_t, f"results/helmholtz_time_{i:04d}.vtu", field_name="Pressure")

    print("Exported harmonic animation.")