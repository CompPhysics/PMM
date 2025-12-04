"""
LMG entanglement entropy: reproduction of Latorre et al. (PhysRevA.71.064101)

Implements:
- LMG Hamiltonian in the fully symmetric (S = N/2) Dicke basis
- Ground state via exact diagonalization
- Reduced density matrix for a block of L spins using Dicke-state combinatorics
- Von Neumann entanglement entropy S_{L,N}
- Plotting routines approximating Figs. 1–5

Dependencies: numpy, matplotlib
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def build_collective_operators(N: int):
    """
    Build S_z, S_+, S_-, S^2 in the S = N/2 symmetric sector.
    Basis: |n> with n = 0..N = number of spins up.
    M = n - N/2 is the S_z eigenvalue.
    """
    S = N / 2.0
    dim = N + 1

    # S_z
    m_vals = np.array([n - S for n in range(dim)], dtype=float)
    Sz = np.diag(m_vals)

    # S_+
    Sp = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        M = n - S
        coef = math.sqrt(S * (S + 1.0) - M * (M + 1.0))
        Sp[n + 1, n] = coef

    # S_- = (S_+)^\dagger
    Sm = Sp.conj().T

    # S^2 = S(S+1) I in this irreducible representation
    S2 = (S * (S + 1.0)) * np.eye(dim, dtype=complex)

    return Sz, Sp, Sm, S2


def build_hamiltonian(N: int, gamma: float, h_field: float,
                      Sz: np.ndarray, Sp: np.ndarray,
                      Sm: np.ndarray, S2: np.ndarray):
    """
    LMG Hamiltonian in the symmetric sector, Eq. (2) of the paper:

        H = -(λ/N)(1+γ)(S^2 - Sz^2 - N/2)
            - 2h Sz
            - (λ/(2N))(1-γ)(S_+^2 + S_-^2)

    We set λ = 1.
    """
    dim = N + 1
    I = np.eye(dim, dtype=complex)

    Sz2 = Sz @ Sz
    Sp2 = Sp @ Sp
    Sm2 = Sm @ Sm

    H = -(1.0 / N) * (1.0 + gamma) * (S2 - Sz2 - (N / 2.0) * I)
    H += -2.0 * h_field * Sz
    H += -(1.0 / (2.0 * N)) * (1.0 - gamma) * (Sp2 + Sm2)

    # Enforce Hermiticity numerically
    H = 0.5 * (H + H.conj().T)
    return H


def ground_state(H: np.ndarray):
    """Return (E0, |ψ0>) for Hamiltonian H."""
    evals, evecs = np.linalg.eigh(H)
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]
    return E0, psi0


# ---------- Hypergeometric decomposition and reduced density matrix ----------

def precompute_hypergeom_probs(N: int, L: int) -> np.ndarray:
    """
    Precompute p_{l|n} = C(L,l) C(N-L, n-l) / C(N, n)
    for l = 0..L, n = 0..N.

    These are the probabilities in Eq. (5) for the Dicke-state
    decomposition used to build the reduced density matrix.
    """
    dim = N + 1
    NL = N - L

    P = np.zeros((L + 1, dim), dtype=float)

    # log binomial using gamma function to avoid overflow
    from math import lgamma, exp

    log_binom_N = np.array(
        [lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)
         for k in range(dim)],
        dtype=float,
    )
    log_binom_L = np.array(
        [lgamma(L + 1) - lgamma(l + 1) - lgamma(L - l + 1)
         for l in range(L + 1)],
        dtype=float,
    )
    log_binom_NL = np.array(
        [lgamma(NL + 1) - lgamma(k + 1) - lgamma(NL - k + 1)
         for k in range(NL + 1)],
        dtype=float,
    )

    for n in range(dim):
        # l can be at most min(L, n), and n-l must be in [0, N-L]
        l_max = min(L, n)
        for l in range(l_max + 1):
            k = n - l
            if k < 0 or k > NL:
                continue
            log_p = log_binom_L[l] + log_binom_NL[k] - log_binom_N[n]
            P[l, n] = exp(log_p)

    return P


def reduced_density_matrix(psi: np.ndarray, N: int, L: int,
                           P: np.ndarray) -> np.ndarray:
    """
    Build the reduced density matrix ρ_A for a block of L spins
    from the symmetric ground state |ψ>.

    Basis on A: |l> with l = 0..L spins up.
    B has N-L spins with basis |m>, m = 0..(N-L).

    Using
        |n> = Σ_l sqrt(p_{l|n}) |l>_A |n-l>_B
    and |ψ> = Σ_n c_n |n>, we can write coefficients
        T_{l,m} = c_{n} sqrt(p_{l|n})   with n = l + m
    so that ρ_A = T T^†.
    """
    NL = N - L
    dimA = L + 1

    T = np.zeros((dimA, NL + 1), dtype=complex)
    for l in range(dimA):
        for m in range(NL + 1):
            n = l + m
            if n > N:
                break
            p = P[l, n]
            if p <= 0.0:
                continue
            T[l, m] = psi[n] * math.sqrt(p)

    rhoA = T @ T.conj().T

    # Normalize and symmetrize
    rhoA = 0.5 * (rhoA + rhoA.conj().T)
    tr = rhoA.trace().real
    if tr != 0.0:
        rhoA /= tr

    return rhoA


def entanglement_entropy(psi: np.ndarray, N: int, L: int,
                         P: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr(ρ log2 ρ) of the block of size L."""
    rhoA = reduced_density_matrix(psi, N, L, P)
    evals = np.linalg.eigvalsh(rhoA)
    evals = evals[evals > 1e-12]
    S = -np.sum(evals * np.log2(evals))
    return float(S.real)


# ---------- Figure generators (approximate reproductions) ----------

def figure1_entropy_surface(N=200, L=None,
                            h_min=0.0, h_max=2.0, n_h=61,
                            g_min=-1.0, g_max=1.0, n_g=61):
    """
    Approximate Fig. 1: S_{L,N}(h, γ) as a color map.

    Paper: N=500, L=125. Those values are heavier but can be used.
    """
    if L is None:
        L = N // 4  # default: L/N = 0.25, same as paper
    Sz, Sp, Sm, S2 = build_collective_operators(N)
    P = precompute_hypergeom_probs(N, L)

    h_vals = np.linspace(h_min, h_max, n_h)
    g_vals = np.linspace(g_min, g_max, n_g)

    S_vals = np.zeros((n_g, n_h))

    for ig, g in enumerate(g_vals):
        print(f"[Fig1] γ = {g:.3f}")
        for ih, h in enumerate(h_vals):
            H = build_hamiltonian(N, g, h, Sz, Sp, Sm, S2)
            _, psi0 = ground_state(H)
            S_vals[ig, ih] = entanglement_entropy(psi0, N, L, P)

    plt.figure(figsize=(6, 4))
    extent = [h_min, h_max, g_min, g_max]
    im = plt.imshow(S_vals, extent=extent, origin="lower",
                    aspect="auto")
    plt.xlabel("h")
    plt.ylabel("γ")
    plt.title(f"Entropy surface (N={N}, L={L})")
    cbar = plt.colorbar(im)
    cbar.set_label("entanglement entropy")
    plt.tight_layout()
    plt.savefig("fig1_entropy_surface.png", dpi=300)


def figure2_entropy_vs_h_gamma0():
    """
    Approximate Fig. 2: S(h) for γ=0 and several (N,L).
    Paper sets:
        N=500,  L=25,75,125,250
        N=1000, L=50,150,250,500
    Here we implement the same, but you can lower N for speed.
    """
    gamma = 0.0
    # adjust N_list if needed for performance
    configs = [
        (500, [25, 75, 125, 250]),
        (1000, [50, 150, 250, 500]),
    ]

    h_vals = np.linspace(0.0, 2.0, 101)
    plt.figure(figsize=(6, 4))

    for N, L_list in configs:
        Sz, Sp, Sm, S2 = build_collective_operators(N)
        for L in L_list:
            print(f"[Fig2] N={N}, L={L}")
            P = precompute_hypergeom_probs(N, L)
            S_curve = []
            for h in h_vals:
                H = build_hamiltonian(N, gamma, h, Sz, Sp, Sm, S2)
                _, psi0 = ground_state(H)
                S_curve.append(entanglement_entropy(psi0, N, L, P))
            S_curve = np.array(S_curve)
            plt.plot(h_vals, S_curve, label=f"N={N}, L={L}")

    plt.xlabel("h")
    plt.ylabel("entanglement entropy")
    plt.title("Entropy vs h at γ=0")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig2_entropy_vs_h_gamma0.png", dpi=300)


def figure3_entropy_near_critical_gamma0():
    """
    Approximate Fig. 3: S(h) near h=1 for γ=0, N=2000, various L.
    """
    gamma = 0.0
    N = 2000
    L_list = [125, 250, 500, 1000]
    h_vals = np.linspace(1.0, 1.1, 51)

    Sz, Sp, Sm, S2 = build_collective_operators(N)

    plt.figure(figsize=(6, 4))
    for L in L_list:
        print(f"[Fig3] N={N}, L={L}")
        P = precompute_hypergeom_probs(N, L)
        S_curve = []
        for h in h_vals:
            H = build_hamiltonian(N, gamma, h, Sz, Sp, Sm, S2)
            _, psi0 = ground_state(H)
            S_curve.append(entanglement_entropy(psi0, N, L, P))
        plt.plot(h_vals, S_curve, label=f"N={N}, L={L}")

    plt.xlabel("h")
    plt.ylabel("entanglement entropy")
    plt.title("Entropy near h=1 (γ=0)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig3_entropy_near_critical_gamma0.png", dpi=300)


def figure4_entropy_vs_L_at_critical():
    """
    Approximate Fig. 4: S(L) at h=1, N=2000, several γ.

    Paper uses γ = 0,0.25,0.5,0.75,1.
    """
    N = 2000
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Sample L values (up to N/2 due to symmetry S(L)=S(N-L))
    L_vals = np.arange(25, N // 2 + 1, 25)

    Sz, Sp, Sm, S2 = build_collective_operators(N)

    plt.figure(figsize=(6, 4))

    for g in gammas:
        print(f"[Fig4] γ={g}")
        # ground state at h=1 for this γ
        H = build_hamiltonian(N, g, 1.0, Sz, Sp, Sm, S2)
        _, psi0 = ground_state(H)

        S_L = []
        for L in L_vals:
            P = precompute_hypergeom_probs(N, L)
            S_L.append(entanglement_entropy(psi0, N, L, P))
        S_L = np.array(S_L)
        plt.plot(L_vals, S_L, label=f"γ={g}")

    plt.xlabel("L")
    plt.ylabel("entanglement entropy")
    plt.title("Entropy vs L at h=1, N=2000")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig4_entropy_vs_L_critical.png", dpi=300)


def figure5_entropy_vs_gamma_at_critical():
    """
    Approximate Fig. 5: S(γ) at h=1, N=2000, L=125,250,500,1000.
    """
    N = 2000
    L_list = [125, 250, 500, 1000]
    gammas = np.linspace(-1.0, 1.0, 81)

    Sz, Sp, Sm, S2 = build_collective_operators(N)

    plt.figure(figsize=(6, 4))

    for L in L_list:
        print(f"[Fig5] L={L}")
        P = precompute_hypergeom_probs(N, L)
        S_gamma = []
        for g in gammas:
            H = build_hamiltonian(N, g, 1.0, Sz, Sp, Sm, S2)
            _, psi0 = ground_state(H)
            S_gamma.append(entanglement_entropy(psi0, N, L, P))
        plt.plot(gammas, S_gamma, label=f"L={L}")

    plt.xlabel("γ")
    plt.ylabel("entanglement entropy")
    plt.title("Entropy vs γ at h=1, N=2000")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig5_entropy_vs_gamma_critical.png", dpi=300)


# ---------- Main ----------

if __name__ == "__main__":
    # WARNING:
    #   The full parameter set from the paper (N up to 2000)
    #   is computationally intensive. You may want to:
    #   - Comment out some figures,
    #   - Start with smaller N for testing (e.g., N=200),
    #   - Reduce grid sizes (n_h, n_g) in figure1_entropy_surface.

    # Example: smaller, quicker test of Fig. 1
    # figure1_entropy_surface(N=200, L=50, n_h=41, n_g=41)

    # Uncomment what you want to generate:

    # figure1_entropy_surface(N=500, L=125, n_h=61, n_g=61)
    # figure2_entropy_vs_h_gamma0()
    # figure3_entropy_near_critical_gamma0()
    # figure4_entropy_vs_L_at_critical()
    # figure5_entropy_vs_gamma_at_critical()

    # For quick sanity test:
    figure1_entropy_surface(N=100, L=25, n_h=31, n_g=31)

