import os
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

# -----------------------------
# Linear algebra utilities
# -----------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],[1, 0]], dtype=complex)
Y  = np.array([[0,-1j],[1j, 0]], dtype=complex)
Z  = np.array([[1, 0],[0,-1]], dtype=complex)

def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def op_on_qubit(single_op, q, L):
    ops = [I2]*L
    ops[q] = single_op
    return kron_all(ops)

def two_body(opA, qA, opB, qB, L):
    ops = [I2]*L
    ops[qA] = opA
    ops[qB] = opB
    return kron_all(ops)

def k_xx_yy(i, a, L):
    return 0.5*(two_body(X,i,X,a,L) + two_body(Y,i,Y,a,L))

# -----------------------------
# Pairing Hamiltonian (qubits)
# -----------------------------
def pairing_hamiltonian_qubits(eps, G):
    """
    H = sum_p eps_p Z_p - (G/4) sum_{p!=q} (X_p X_q + Y_p Y_q) + const (omitted)
    """
    L = len(eps)
    dim = 2**L
    H = np.zeros((dim, dim), dtype=complex)

    for p in range(L):
        H += eps[p] * op_on_qubit(Z, p, L)

    for p in range(L):
        for q in range(L):
            if p == q:
                continue
            H += -(G/4) * (two_body(X,p,X,q,L) + two_body(Y,p,Y,q,L))
    return H

def projector_fixed_weight(L, Npairs):
    dim = 2**L
    diag = np.zeros(dim, dtype=complex)
    for idx in range(dim):
        if idx.bit_count() == Npairs:
            diag[idx] = 1.0
    return np.diag(diag)

def restrict_to_projector(A, P):
    """Return restricted matrix A_red on support of projector P."""
    keep = np.where(np.abs(np.diag(P)) > 0.5)[0]
    Ared = A[np.ix_(keep, keep)]
    return Ared, keep

def exact_ground_state(H, P):
    Hred, keep = restrict_to_projector(H, P)
    evals, evecs = la.eigh(Hred)
    psi0_red = evecs[:, 0]
    psi0 = np.zeros(H.shape[0], dtype=complex)
    psi0[keep] = psi0_red
    psi0 /= la.norm(psi0)
    return float(np.real(evals[0])), psi0, keep

# -----------------------------
# pUCCD state preparation
# -----------------------------
def reference_state(L, Npairs):
    idx = sum(1 << q for q in range(Npairs))  # qubits 0..Npairs-1 set to 1
    psi = np.zeros(2**L, dtype=complex)
    psi[idx] = 1.0
    return psi

def uccd_trotter_state(L, Npairs, theta_vec, pairs_list, r=1):
    """
    theta_vec: ndarray of length len(pairs_list)
    pairs_list: list of (i,a) pairs, i in occ, a in virt
    r: number of trotter steps
    """
    psi = reference_state(L, Npairs)
    for _ in range(r):
        for t, (i,a) in zip(theta_vec, pairs_list):
            K = k_xx_yy(i, a, L)
            U = la.expm(-1j * (t/r) * K)
            psi = U @ psi
    return psi / la.norm(psi)

def energy(psi, H):
    return float(np.real(np.vdot(psi, H @ psi)))

def fidelity(psi, phi):
    return float(np.abs(np.vdot(psi, phi))**2)

# -----------------------------
# Optimization
# -----------------------------
def optimize_for_r(H, P, E0, psi0, keep, L, Npairs, pairs_list, r,
                   n_starts=12, theta_scale=0.5, maxiter=1200, seed=0):
    """
    Multi-start Nelder–Mead on restricted sector objective:
    minimize E(theta) = <psi(theta)|H|psi(theta)>
    """
    rng = np.random.default_rng(seed)

    Hred, keep2 = restrict_to_projector(H, P)
    assert np.all(keep2 == keep)

    def objective(theta_vec):
        psi = uccd_trotter_state(L, Npairs, theta_vec, pairs_list, r=r)
        # restrict to fixed-weight sector
        psi_red = psi[keep]
        psi_red /= la.norm(psi_red)
        return float(np.real(np.vdot(psi_red, Hred @ psi_red)))

    best = (1e99, None)

    for s in range(n_starts):
        x0 = theta_scale * rng.normal(size=len(pairs_list))
        res = opt.minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-8, "disp": False})
        if res.fun < best[0]:
            best = (res.fun, res.x)

    bestE, bestTheta = best
    psi_best = uccd_trotter_state(L, Npairs, bestTheta, pairs_list, r=r)
    # project and renormalize
    psi_best_red = psi_best[keep]
    psi_best_red /= la.norm(psi_best_red)
    # embed back into full space for fidelity with psi0
    psi_best_full = np.zeros(2**L, dtype=complex)
    psi_best_full[keep] = psi_best_red
    psi_best_full /= la.norm(psi_best_full)

    F = fidelity(psi_best_full, psi0)
    gap = bestE - E0
    return bestE, gap, F, bestTheta

# -----------------------------
# Main benchmark + plotting
# -----------------------------
def main():
    # Problem setup (requested)
    L = 10
    Npairs = 5
    eps = [float(p) for p in range(L)]  # equally spaced
    G = 1.0

    os.makedirs("figures", exist_ok=True)

    H = pairing_hamiltonian_qubits(eps, G)
    P = projector_fixed_weight(L, Npairs)

    E0, psi0, keep = exact_ground_state(H, P)
    print(f"Exact ground energy (fixed-weight sector): {E0:+.10f}")
    print(f"Sector dimension = C(L,Npairs) = {len(keep)} (out of 2^L = {2**L})")

    occ = list(range(Npairs))
    virt = list(range(Npairs, L))
    pairs_list = [(i,a) for i in occ for a in virt]  # 25 parameters for L=10, N=5

    r_list = [1, 2, 3, 4, 6, 8, 10]
    E_list, gap_list, F_list = [], [], []

    # Warm-start: reuse previous optimum as one of the initial points by nudging the search scale
    prev_theta = None

    for r in r_list:
        bestE, gap, F, theta = optimize_for_r(
            H, P, E0, psi0, keep, L, Npairs, pairs_list, r,
            n_starts=10, theta_scale=0.6, maxiter=1400, seed=100+r
        )
        prev_theta = theta
        E_list.append(bestE)
        gap_list.append(gap)
        F_list.append(F)

        print(f"r={r:2d}  E={bestE:+.10f}  gap={gap:+.3e}  F={F:.6f}")

    # Print LaTeX table rows
    print("\nLaTeX table rows (paste into main.tex Table):")
    for r, E, gap, F in zip(r_list, E_list, gap_list, F_list):
        # Scientific format for gap
        gap_str = f"{gap:.2e}".replace("e", "\\times 10^{") + "}"
        # e.g. 1.23\times 10^{-4}
        gap_str = gap_str.replace("+", "")
        gap_str = gap_str.replace("\\times 10^{0}", "\\times 10^{0}")
        print(f"{r} & {E:+.6f} & {gap_str} & {F:.6f} \\\\")

    # -----------------------------
    # Plots saved to PDF (no explicit colors)
    # -----------------------------
    # Energy gap vs r
    plt.figure()
    plt.plot(r_list, gap_list, marker="o")
    plt.xlabel("Trotter steps r")
    plt.ylabel("Energy error ΔE(r) = E_pUCCD(r) - E0")
    plt.title("pUCCD-Trotter energy error vs r (L=10, N=5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/energy_vs_r.pdf")

    # Fidelity vs r
    plt.figure()
    plt.plot(r_list, F_list, marker="o")
    plt.xlabel("Trotter steps r")
    plt.ylabel("Fidelity F(r)")
    plt.title("pUCCD-Trotter fidelity vs r (L=10, N=5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/fidelity_vs_r.pdf")

    print("\nSaved PDFs:")
    print(" - figures/energy_vs_r.pdf")
    print(" - figures/fidelity_vs_r.pdf")

if __name__ == "__main__":
    main()
