import numpy as np
import matplotlib.pyplot as plt

from fem_engine.mesh import create_rectangle_mesh, FunctionSpace, MixedFunctionSpace
from fem_engine.element import Quad4
from fem_engine.assembler import MixedAssembler
from fem_engine.fem_solver import solve_linear
from fem_engine.postprocess import evaluate_field_at_point

# Physical parameters
L = 1.0
W = 1.0

c = 343.0
rho0 = 1.225

# Gaussian monopole Source
def q_source(x, y):
    sigma = 0.03 # Gaussian width
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    amplitude = 1.0 / (2.0 * np.pi * sigma**2)

    return amplitude * np.exp(-r2 / (2.0 * sigma**2))


# Frequency-dependent weak form

def make_helmholtz_form(frequency):

    omega = 2*np.pi*frequency
    k = omega/c


    def helmholtz_weak(mapped, pos_gp, e):

        (N_r, B_r, p_r, grad_pr) = mapped[0]
        (N_i, B_i, p_i, grad_pi) = mapped[1]

        x, y = pos_gp

        q = 0.0005 * q_source(x,y)

        # Real part
        # lap(pr) + k^2 pr = 0

        stiffness_r = B_r.T @ grad_pr
        mass_r = -np.outer(N_r, k**2 * p_r)

        R_r = stiffness_r + mass_r

        # Imaginary part
        # lap(pi) + k^2 pi = -omega * rho * Q

        stiffness_i = B_i.T @ grad_pi
        mass_i = -np.outer(N_i, k**2 * p_i)

        source = -np.outer(N_i, omega * rho0 * q)

        R_i = stiffness_i + mass_i + source

        return [R_r,R_i]


    return helmholtz_weak

if __name__ == "__main__":

    mesh = create_rectangle_mesh(L=L, W=W, n_x=60, n_y=60, x0=0, y0=0)

    V_r = FunctionSpace(mesh, Quad4(), n_components=1)
    V_i = FunctionSpace(mesh, Quad4(), n_components=1)

    V = MixedFunctionSpace([V_r, V_i])

    # Rigid walls:
    # del p/del n = 0
    # This is the natural BC of the weak form,
    # therefore NO boundary conditions are applied.

    def apply_bcs(R,K,U):
        return R,K

    # Microphone location
    mic = np.array([0.2,0.2])

    # Frequency sweep
    frequencies = np.linspace(10, 2000, 200)

    SPL = []

    # Sweep

    for frequency in frequencies:


        print(f"Solving {frequency:.1f} Hz")

        assembler = MixedAssembler(V, make_helmholtz_form(frequency), quad_degree=2)

        # 1. Initialize U = 0
        U0 = np.zeros(V.ndofs)

        # 2. Assemble the global matrices at U = 0
        R_global, K_global = assembler.assemble(U0)

        # 3. Extract the load vector (F = -R)
        F_global = -R_global

        # 5. Solve the linear system
        print("--- Solving Helmholtz ---")
        U = solve_linear(K_global, F_global)

        p_real, p_imag = V.split(U)

        # Microphone evaluation

        Pr = evaluate_field_at_point(mesh, V_r, p_real, mic)
        Pi = evaluate_field_at_point(mesh,V_i, p_imag, mic)

        # remove possible array wrapping
        Pr = Pr[0] if np.ndim(Pr)>0 else Pr
        Pi = Pi[0] if np.ndim(Pi)>0 else Pi

        # complex magnitude
        Pmag = np.sqrt(Pr**2 + Pi**2)

        # SPL
        pref = 2e-5

        spl = 20*np.log10(Pmag/pref)


        SPL.append(spl)

    # Plot spectrum

    SPL = np.array(SPL)

    plt.figure(figsize=(9,5))

    plt.plot(frequencies, SPL, linewidth=2)

    plt.xlabel("Frequency [Hz]")

    plt.ylabel("Sound Pressure Level [dB]")

    plt.title("Helmholtz cavity response at microphone (0.2,0.2)")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/microphone_SPL.png", dpi=300)

    plt.show()