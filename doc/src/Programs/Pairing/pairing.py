"""
Lipkin model (with optional symmetric J = N/2 sector).
Pairing Hamiltonian in the seniority-zero subspace.
For each model: 
Build exact spectra (lowest k levels vs control param).
Train a big PMM (same dimension as physical Hilbert space).
Train a reduced PMM (dimension n_eff < dim_phys).

PMM training uses analytic gradients of eigenvalues w.r.t. the Hamiltonian (no auto-diff).


I keep the PMM matrices real symmetric, which simplifies gradients.
"""
import math
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

# ============================================================
# 1. LIPKIN: Collective spin operators
#    (A) Full 2^N Hilbert space
#    (B) Symmetric J = N/2 sector (block-diagonalized)
# ============================================================

def build_collective_operators_full(N):
    """
    Build Jx, Jy, Jz in the full 2^N Hilbert space
    from tensor products of single-qubit Pauli matrices.
    """
    dim = 2**N
    sx = np.array([[0, 1],
                   [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],
                   [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],
                   [0, -1]], dtype=np.complex128)
    id2 = np.eye(2, dtype=np.complex128)

    Jx = np.zeros((dim, dim), dtype=np.complex128)
    Jy = np.zeros((dim, dim), dtype=np.complex128)
    Jz = np.zeros((dim, dim), dtype=np.complex128)

    for site in range(N):
        ops = []
        for pos in range(N):
            if pos == site:
                ops.append((sx, sy, sz))
            else:
                ops.append((id2, id2, id2))

        sx_i = ops[0][0]
        sy_i = ops[0][1]
        sz_i = ops[0][2]
        for pos in range(1, N):
            sx_i = np.kron(sx_i, ops[pos][0])
            sy_i = np.kron(sy_i, ops[pos][1])
            sz_i = np.kron(sz_i, ops[pos][2])

        Jx += 0.5 * sx_i
        Jy += 0.5 * sy_i
        Jz += 0.5 * sz_i

    return Jx, Jy, Jz


def build_collective_operators_symmetric(N):
    """
    Build Jx, Jy, Jz in the fully symmetric total-spin sector J = N/2.

    Basis: |J, M>, M = J, J-1, ..., -J. Dimension = N+1.
    """
    J = N / 2.0
    dim = int(2 * J + 1)

    M_vals = np.arange(J, -J - 1, -1, dtype=float)

    Jp = np.zeros((dim, dim), dtype=np.complex128)
    Jm = np.zeros((dim, dim), dtype=np.complex128)
    Jz = np.zeros((dim, dim), dtype=np.complex128)

    for i, M in enumerate(M_vals):
        # J_z |J,M> = M |J,M>
        Jz[i, i] = M

        # J_+ |J,M> -> |J,M+1>, index i-1
        if i > 0:
            coef = math.sqrt(J * (J + 1.0) - M * (M + 1.0))
            Jp[i - 1, i] = coef

        # J_- |J,M> -> |J,M-1>, index i+1
        if i < dim - 1:
            coef = math.sqrt(J * (J + 1.0) - M * (M - 1.0))
            Jm[i + 1, i] = coef

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)

    return Jx, Jy, Jz


def lipkin_from_collective(Jx, Jy, Jz, epsilon, V, N):
    """
    Lipkin Hamiltonian:
        H = epsilon * Jz + (V/N) * (Jx^2 - Jy^2)
    """
    return epsilon * Jz + (V / N) * (Jx @ Jx - Jy @ Jy)


def generate_lipkin_spectrum(Jx, Jy, Jz, epsilon, N,
                             V_train, V_test, n_levels=3):
    """
    Generate lowest n_levels eigenvalues for Lipkin model
    (training and test sets) in the chosen representation.
    """
    from numpy.linalg import eigvalsh

    H0 = epsilon * Jz
    Hint = (1.0 / N) * (Jx @ Jx - Jy @ Jy)

    E_train = []
    for V in V_train:
        H = H0 + V * Hint
        evals = eigvalsh(H)
        E_train.append(evals[:n_levels])
    E_train = np.array(E_train)

    E_test = []
    for V in V_test:
        H = H0 + V * Hint
        evals = eigvalsh(H)
        E_test.append(evals[:n_levels])
    E_test = np.array(E_test)

    return E_train, E_test, H0, Hint


# ============================================================
# 2. PAIRING HAMILTONIAN in seniority-zero subspace
# ============================================================

def build_pairing_basis(omega, N_pairs):
    """
    Seniority-zero basis for the pairing model:
      - omega doubly-degenerate levels j=0..omega-1
      - each level holds either 0 or 1 pair
      - N_pairs total pairs

    Basis states are represented by sorted tuples of occupied levels,
    e.g. (0,2) means pairs on levels 0 and 2.
    """
    levels = range(omega)
    basis = list(combinations(levels, N_pairs))
    cfg_to_idx = {cfg: i for i, cfg in enumerate(basis)}
    return basis, cfg_to_idx


def pairing_hamiltonian(omega, N_pairs, eps_levels, G):
    """
    Pairing Hamiltonian in seniority-zero subspace:

        H = sum_j 2 eps_j n_j - G sum_{i,j} P_i^† P_j

    where n_j = 0 or 1 is pair occupation on level j,
    and P_i^† P_j with i != j moves a pair from level j to i.
    """
    basis, cfg_to_idx = build_pairing_basis(omega, N_pairs)
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.complex128)

    for a, cfg in enumerate(basis):
        n_j = np.zeros(omega, dtype=int)
        for j in cfg:
            n_j[j] = 1

        # Diagonal: sum_j 2 eps_j n_j
        diag_sp = np.sum(2.0 * eps_levels * n_j)

        # Diagonal: -G sum_i P_i^† P_i = -G * (# of pairs)
        diag_pair = -G * np.sum(n_j)

        H[a, a] += diag_sp + diag_pair

        # Off-diagonal: move pair j -> i
        for j in cfg:
            for i in range(omega):
                if i == j:
                    continue
                if n_j[i] == 0:
                    new_cfg = tuple(sorted((set(cfg) - {j}) | {i}))
                    b = cfg_to_idx[new_cfg]
                    H[b, a] += -G

    return H


def generate_pairing_spectrum(omega, N_pairs, eps_levels,
                              G_train, G_test, n_levels=3):
    """
    Generate lowest n_levels eigenvalues for pairing model
    (training and test sets) as functions of G.
    """
    from numpy.linalg import eigvalsh

    basis, _ = build_pairing_basis(omega, N_pairs)
    dim = len(basis)
    print(f"Pairing model: omega={omega}, N_pairs={N_pairs}, dim={dim}")

    E_train = []
    for G in G_train:
        H = pairing_hamiltonian(omega, N_pairs, eps_levels, G)
        evals = eigvalsh(H)
        E_train.append(evals[:n_levels])
    E_train = np.array(E_train)

    E_test = []
    for G in G_test:
        H = pairing_hamiltonian(omega, N_pairs, eps_levels, G)
        evals = eigvalsh(H)
        E_test.append(evals[:n_levels])
    E_test = np.array(E_test)

    return E_train, E_test, dim


# ============================================================
# 3. PMM model M(c) = A + c B (real symmetric, spectrum only)
# ============================================================

def init_pmm_params(n, seed=0, scale=0.1):
    """
    Initialize real symmetric A, B.
    """
    rng = np.random.default_rng(seed)
    A = scale * rng.normal(size=(n, n))
    B = scale * rng.normal(size=(n, n))
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    return A, B


def pmm_hamiltonian(A, B, c):
    """
    Real symmetric PMM Hamiltonian M(c) = A + c B.
    """
    H = A + c * B
    return 0.5 * (H + H.T)


def pmm_eigs_for_c(A, B, c, k):
    """
    Return lowest k eigenvalues and eigenvectors of M(c).
    """
    from numpy.linalg import eigh
    H = pmm_hamiltonian(A, B, c)
    evals, evecs = eigh(H)
    return evals[:k], evecs[:, :k]


def pmm_eigs_for_param_array(A, B, param_array, k):
    """
    Evaluate eigenvalues for many control-parameter values.
    Returns array of shape (len(param_array), k).
    """
    vals = []
    for p in param_array:
        e, _ = pmm_eigs_for_c(A, B, p, k)
        vals.append(e)
    return np.array(vals)


# ============================================================
# 4. Loss + training for spectrum (big & reduced PMM)
#    analytic gradients for eigenvalues
# ============================================================

def adam_update(param, grad_param, m, v, t, lr=1e-2,
                beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad_param
    v = beta2 * v + (1 - beta2) * (grad_param * grad_param)
    m_hat = m / (1 - beta1 ** (t + 1))
    v_hat = v / (1 - beta2 ** (t + 1))
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v


def train_spectrum(A, B, params_train, E_train,
                   lr=1e-2, n_iter=2000, label="Spectrum"):
    """
    Train PMM on lowest k eigenvalues as a function of params_train.
    A, B: real symmetric (n, n)
    params_train: array of control parameters, shape (n_points,)
    E_train: true eigenvalues, shape (n_points, k)
    """
    A = A.copy()
    B = B.copy()
    n = A.shape[0]
    n_points, k = E_train.shape

    mA = np.zeros_like(A)
    vA = np.zeros_like(A)
    mB = np.zeros_like(B)
    vB = np.zeros_like(B)

    for t in range(n_iter):
        grad_A = np.zeros_like(A)
        grad_B = np.zeros_like(B)
        loss_val = 0.0

        for idx, p in enumerate(params_train):
            # PMM eigensystem at param p
            evals, evecs = pmm_eigs_for_c(A, B, p, k)
            diff = evals - E_train[idx]   # shape (k,)

            loss_val += np.sum(diff**2)

            # Gradient of eigenvalue λ_l wrt H is v_l v_l^T (for normalized real v_l)
            # Loss = (1/(n_points*k)) sum (λ - E_true)^2
            # => ∂Loss/∂H = (2/(n_points*k)) sum (λ - E_true) v v^T
            factor = 2.0 / (n_points * k)
            for l in range(k):
                g_coeff = factor * diff[l]
                v = evecs[:, l]  # (n,)
                outer = np.outer(v, v)  # symmetric
                grad_A += g_coeff * outer
                grad_B += g_coeff * p * outer

        loss_val /= (n_points * k)

        # Adam update
        A, mA, vA = adam_update(A, grad_A, mA, vA, t, lr=lr)
        B, mB, vB = adam_update(B, grad_B, mB, vB, t, lr=lr)

        # Re-symmetrize (numerical safety)
        A = 0.5 * (A + A.T)
        B = 0.5 * (B + B.T)

        if t % 200 == 0 or t == n_iter - 1:
            print(f"[{label}] iter {t:5d}  loss = {loss_val:.6e}")

    return A, B


# ============================================================
# 5. Main: Lipkin + Pairing, big PMM + reduced PMM (no JAX)
# ============================================================

if __name__ == "__main__":
    # ------------------------
    # LIPKIN SETTINGS
    # ------------------------
    N = 8               # number of spins
    epsilon_lip = 1.0
    k_levels = 3        # ground + 2 excited

    USE_SYMMETRIC_SECTOR = True  # True: J=N/2 block (dim=N+1); False: full 2^N

    V_train = np.linspace(0.0, 2.0, 20)
    V_test  = np.linspace(0.0, 2.0, 100)

    # Build Jx, Jy, Jz in chosen representation
    if USE_SYMMETRIC_SECTOR:
        Jx_np, Jy_np, Jz_np = build_collective_operators_symmetric(N)
        dim_lip = Jx_np.shape[0]  # N+1
        print(f"Lipkin: symmetric J = N/2 sector, dim = {dim_lip}")
    else:
        Jx_np, Jy_np, Jz_np = build_collective_operators_full(N)
        dim_lip = Jx_np.shape[0]  # 2^N
        print(f"Lipkin: full Hilbert space, dim = {dim_lip}")

    # Exact Lipkin spectrum
    E_train_lip, E_test_lip, H0_lip, Hint_lip = generate_lipkin_spectrum(
        Jx_np, Jy_np, Jz_np, epsilon_lip, N, V_train, V_test, n_levels=k_levels
    )

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(V_test, E_test_lip[:, ell], lw=2, label=f"Exact Lipkin E_{ell}")
    plt.scatter(V_train, E_train_lip[:, 0], color="k", s=40,
                label="Train points (ground)")
    plt.xlabel("V")
    plt.ylabel("E_n(V)")
    plt.title(f"Lipkin spectrum (N={N}, dim={dim_lip})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Big PMM for Lipkin -----
    n_pmm_big_lip = dim_lip
    A0_big_lip, B0_big_lip = init_pmm_params(n_pmm_big_lip, seed=123)

    A_big_lip, B_big_lip = train_spectrum(A0_big_lip, B0_big_lip,
                                          V_train, E_train_lip,
                                          lr=1e-2, n_iter=2000,
                                          label="Lipkin big PMM")

    E_pred_big_lip = pmm_eigs_for_param_array(A_big_lip, B_big_lip,
                                              V_test, k_levels)

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(V_test, E_test_lip[:, ell], lw=2, label=f"Exact Lipkin E_{ell}")
        plt.plot(V_test, E_pred_big_lip[:, ell], "--", lw=2,
                 label=f"Big PMM E_{ell}")
    plt.xlabel("V")
    plt.ylabel("E_n(V)")
    plt.title(f"Lipkin: big PMM vs exact (dim={dim_lip})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Reduced PMM for Lipkin -----
    n_eff_lip = min(6, dim_lip)
    print(f"\nLipkin: training reduced PMM with n_eff = {n_eff_lip} "
          f"for lowest {k_levels} levels.")
    A0_eff_lip, B0_eff_lip = init_pmm_params(n_eff_lip, seed=456)

    A_eff_lip, B_eff_lip = train_spectrum(A0_eff_lip, B0_eff_lip,
                                          V_train, E_train_lip,
                                          lr=5e-3, n_iter=2000,
                                          label="Lipkin reduced PMM")

    E_pred_eff_lip = pmm_eigs_for_param_array(A_eff_lip, B_eff_lip,
                                              V_test, k_levels)

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(V_test, E_test_lip[:, ell], lw=2, label=f"Exact Lipkin E_{ell}")
        plt.plot(V_test, E_pred_eff_lip[:, ell], "--", lw=2,
                 label=f"Reduced PMM E_{ell}")
    plt.xlabel("V")
    plt.ylabel("E_n(V)")
    plt.title(f"Lipkin: reduced PMM vs exact (n_eff={n_eff_lip}, dim={dim_lip})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # PAIRING MODEL EXAMPLE
    # --------------------------------------------------------
    omega = 4        # number of doubly-degenerate levels
    N_pairs = 2      # number of pairs
    eps_levels = np.linspace(-1.5, 1.5, omega)  # single-particle energies

    G_train = np.linspace(0.0, 1.0, 20)   # pairing strength grid
    G_test  = np.linspace(0.0, 1.0, 100)

    E_train_pair, E_test_pair, dim_pair = generate_pairing_spectrum(
        omega, N_pairs, eps_levels,
        G_train, G_test, n_levels=k_levels
    )

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(G_test, E_test_pair[:, ell], lw=2, label=f"Exact Pairing E_{ell}")
    plt.scatter(G_train, E_train_pair[:, 0], color="k", s=40,
                label="Train points (ground)")
    plt.xlabel("G")
    plt.ylabel("E_n(G)")
    plt.title(f"Pairing spectrum (omega={omega}, N_pairs={N_pairs}, dim={dim_pair})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Big PMM for Pairing -----
    n_pmm_big_pair = dim_pair
    A0_big_pair, B0_big_pair = init_pmm_params(n_pmm_big_pair, seed=789)

    A_big_pair, B_big_pair = train_spectrum(A0_big_pair, B0_big_pair,
                                            G_train, E_train_pair,
                                            lr=1e-2, n_iter=2000,
                                            label="Pairing big PMM")

    E_pred_big_pair = pmm_eigs_for_param_array(A_big_pair, B_big_pair,
                                               G_test, k_levels)

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(G_test, E_test_pair[:, ell], lw=2, label=f"Exact Pairing E_{ell}")
        plt.plot(G_test, E_pred_big_pair[:, ell], "--", lw=2,
                 label=f"Big PMM E_{ell}")
    plt.xlabel("G")
    plt.ylabel("E_n(G)")
    plt.title(f"Pairing: big PMM vs exact (dim={dim_pair})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- Reduced PMM for Pairing -----
    n_eff_pair = min(6, dim_pair)
    print(f"\nPairing: training reduced PMM with n_eff = {n_eff_pair} "
          f"for lowest {k_levels} levels.")
    A0_eff_pair, B0_eff_pair = init_pmm_params(n_eff_pair, seed=101112)

    A_eff_pair, B_eff_pair = train_spectrum(A0_eff_pair, B0_eff_pair,
                                            G_train, E_train_pair,
                                            lr=5e-3, n_iter=2000,
                                            label="Pairing reduced PMM")

    E_pred_eff_pair = pmm_eigs_for_param_array(A_eff_pair, B_eff_pair,
                                               G_test, k_levels)

    plt.figure(figsize=(8, 5))
    for ell in range(k_levels):
        plt.plot(G_test, E_test_pair[:, ell], lw=2, label=f"Exact Pairing E_{ell}")
        plt.plot(G_test, E_pred_eff_pair[:, ell], "--", lw=2,
                 label=f"Reduced PMM E_{ell}")
    plt.xlabel("G")
    plt.ylabel("E_n(G)")
    plt.title(f"Pairing: reduced PMM vs exact (n_eff={n_eff_pair}, dim={dim_pair})")
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
Representations 
USE_SYMMETRIC_SECTOR = True → Lipkin lives in the J=N/2 block (dimension N+1).
False → full 2^N Hilbert space.

Control parameters 
Lipkin: V_train, V_test.
Pairing: G_train, G_test.

Model sizes 
k_levels: how many low-lying levels to fit.
n_eff_lip, n_eff_pair: size of reduced PMMs (effective models).

Training 
n_iter and lr in train_spectrum for big and reduced PMMs.
"""
