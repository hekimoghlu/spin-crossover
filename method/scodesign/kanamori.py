"""Kanamori d6 model, seniority-zero hard-core-boson Hamiltonian (manuscript sec. 3.2-3.3, app. B).

Conventions
-----------
* five orbitals m = 0,1,2 (t2g, eps = -0.4 Delta) and m = 3,4 (eg, eps = +0.6 Delta)
* U' = 0, U = 2 J_H, J_H = P/4  (manuscript choice)
* qubit m <-> orbital m; computational basis index x has qubit q on bit (4-q),
  so the reference |11100> is x = 0b11100 = 28; occupied pair <-> |1>, n_m = (1 - Z_m)/2.
"""
import itertools
import numpy as np

NQ = 5
DIM = 1 << NQ
REF_INDEX = 0b11100
BLOCK_STATES = np.array([x for x in range(DIM) if bin(x).count("1") == 3])  # 10 states
PAIRS = [(m, mp) for m in range(NQ) for mp in range(m + 1, NQ)]          # 10 pairs


def bit(x, q):
    return (x >> (NQ - 1 - q)) & 1


def orbital_energies(Delta):
    Delta = np.asarray(Delta, dtype=float)
    return np.stack([-0.4 * Delta] * 3 + [0.6 * Delta] * 2, axis=-1)


def hb_params(Delta, J):
    """Return (eps[...,5], U, W, Jhop) with U' = 0, U = 2J, W = 4U' - 2J."""
    eps = orbital_energies(Delta)
    J = np.asarray(J, dtype=float)
    U = 2.0 * J
    Up = 0.0
    W = 4.0 * Up - 2.0 * J
    return eps, U, W, J


# ------------------------------------------------------------------ block Hamiltonian
def hb_block(Delta, J):
    """Ten-dimensional three-pair block of H_hb, batched over leading axis."""
    Delta = np.atleast_1d(np.asarray(Delta, float))
    J = np.atleast_1d(np.asarray(J, float))
    eps, U, W, Jh = hb_params(Delta, J)
    n = len(Delta)
    H = np.zeros((n, 10, 10))
    occ = np.array([[bit(x, q) for q in range(NQ)] for x in BLOCK_STATES], float)  # (10,5)
    onsite = 2.0 * eps + U[:, None]                                           # (n,5)
    npairs = np.array([sum(o[m] * o[mp] for m, mp in PAIRS) for o in occ])       # (10,)
    diag = onsite @ occ.T + W[:, None] * npairs[None, :]
    H[:, np.arange(10), np.arange(10)] = diag
    for a, xa in enumerate(BLOCK_STATES):
        for b, xb in enumerate(BLOCK_STATES):
            if bin(xa ^ xb).count("1") == 2 and a != b:
                H[:, a, b] = Jh
    return H


def exact_ls(Delta, J):
    """Lowest eigenvalue of the three-pair block (the low-spin energy)."""
    H = hb_block(Delta, J)
    return np.linalg.eigvalsh(H)[:, 0]


def exact_ls_vec(Delta, J):
    H = hb_block(Delta, J)
    w, v = np.linalg.eigh(H)
    return w[:, 0], v[:, :, 0]


def e_ref(Delta, J):
    """E(t2g^6) = -2.4 Delta + 6 J - 6 J."""
    Delta = np.asarray(Delta, float); J = np.asarray(J, float)
    return -2.4 * Delta + 3 * 2.0 * J + 3 * (-2.0 * J)


def e_hs(Delta, J):
    """Closed-form M_S = 2 determinant t2g^4 eg^2: -0.4 Delta + 2J - 10J."""
    Delta = np.asarray(Delta, float); J = np.asarray(J, float)
    return -0.4 * Delta + 2.0 * J - 10.0 * J


def dH_mf(Delta, J):
    return e_hs(Delta, J) - e_ref(Delta, J)  # = 2 Delta - 8 J


def e_pt2(Delta, J):
    """Second-order pair-correlation energy, eq. (pt2), summed over the six single pair excitations."""
    H = hb_block(Delta, J)
    r = int(np.where(BLOCK_STATES == REF_INDEX)[0][0])
    e0 = H[:, r, r]
    tot = np.zeros(H.shape[0])
    for s in range(10):
        if s == r:
            continue
        v = H[:, s, r]
        tot -= v ** 2 / (H[:, s, s] - e0)
    return tot


# ------------------------------------------------------------------ Pauli decomposition
def pauli_terms(Delta, J):
    """Return list of (pauli_string, coeff) for one complex: 5 Z, 10 ZZ, 10 XX, 10 YY (+ c0)."""
    eps, U, W, Jh = hb_params(float(Delta), float(J))
    eps = np.asarray(eps).ravel()
    c0 = 0.5 * np.sum(2 * eps + U) + 10.0 / 4.0 * W
    terms = []
    for m in range(NQ):
        cm = -0.5 * (2 * eps[m] + U) - W
        s = ["I"] * NQ; s[m] = "Z"
        terms.append(("".join(s), float(cm)))
    for m, mp in PAIRS:
        s = ["I"] * NQ; s[m] = "Z"; s[mp] = "Z"
        terms.append(("".join(s), float(W / 4.0)))
    for P in "XY":
        for m, mp in PAIRS:
            s = ["I"] * NQ; s[m] = P; s[mp] = P
            terms.append(("".join(s), float(Jh / 2.0)))
    return float(c0), terms


def pauli_coeff_arrays(Delta, J):
    """Batched coefficients: cZ (n,5), cZZ (n,10), cXX (n,10) (= cYY), c0 (n,)."""
    Delta = np.atleast_1d(np.asarray(Delta, float)); J = np.atleast_1d(np.asarray(J, float))
    eps, U, W, Jh = hb_params(Delta, J)
    cZ = -0.5 * (2 * eps + U[:, None]) - W[:, None]
    cZZ = np.repeat((W / 4.0)[:, None], 10, axis=1)
    cXX = np.repeat((Jh / 2.0)[:, None], 10, axis=1)
    c0 = 0.5 * np.sum(2 * eps + U[:, None], axis=1) + 2.5 * W
    return cZ, cZZ, cXX, c0


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(s):
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, _PAULI[ch])
    return M


def hamiltonian_matrix(Delta, J):
    c0, terms = pauli_terms(Delta, J)
    H = c0 * np.eye(DIM, dtype=complex)
    for s, c in terms:
        H = H + c * pauli_matrix(s)
    return H


def measurement_groups(terms):
    """Group the 35 terms into the three commuting product bases Z, X, Y."""
    groups = {"Z": [], "X": [], "Y": []}
    for s, c in terms:
        kinds = set(ch for ch in s if ch != "I")
        assert len(kinds) == 1
        groups[kinds.pop()].append((s, c))
    return groups


# ------------------------------------------------------------------ full ten-mode space
def _apply(ops, state):
    """Apply fermion operators (list, rightmost first) to an occupation bitmask. Return (sign, new) or None."""
    sign = 1
    st = state
    for mode, dag in reversed(ops):
        occ = (st >> mode) & 1
        if dag and occ:
            return None
        if (not dag) and not occ:
            return None
        # JW sign: number of occupied modes with index < mode
        nb = bin(st & ((1 << mode) - 1)).count("1")
        if nb & 1:
            sign = -sign
        st ^= (1 << mode)
    return sign, st


def full_space_matrices(Delta, J, spin_flip_sign=+1.0, n_el=6):
    """Kanamori H in the full N=6 Fock space of 10 spin orbitals, plus S^2 and S_z.

    mode(m, sigma) = 2m + sigma, sigma = 0 (up), 1 (down).
    spin_flip_sign = +1 reproduces the manuscript's eq. (kanamori) as printed.
    """
    eps = orbital_energies(Delta)
    U = 2.0 * J; Up = U - 2.0 * J; Jh = J
    basis = [s for s in range(1 << 10) if bin(s).count("1") == n_el]
    idx = {s: i for i, s in enumerate(basis)}
    D = len(basis)
    H = np.zeros((D, D)); S2 = np.zeros((D, D)); Sz = np.zeros(D)
    md = lambda m, s: 2 * m + s

    def add(Mat, ops, coef):
        for i, st in enumerate(basis):
            r = _apply(ops, st)
            if r is None:
                continue
            sg, new = r
            Mat[idx[new], i] += coef * sg

    for i, st in enumerate(basis):
        n = [[(st >> md(m, s)) & 1 for s in (0, 1)] for m in range(5)]
        e = 0.0
        for m in range(5):
            e += eps[m] * (n[m][0] + n[m][1]) + U * n[m][0] * n[m][1]
        for m in range(5):
            for mp in range(m + 1, 5):
                for s in (0, 1):
                    for sp in (0, 1):
                        e += Up * n[m][s] * n[mp][sp]
                    e -= Jh * n[m][s] * n[mp][s]
        H[i, i] = e
        Sz[i] = 0.5 * sum(n[m][0] - n[m][1] for m in range(5))
    for m in range(5):
        for mp in range(5):
            if m == mp:
                continue
            add(H, [(md(m, 0), True), (md(m, 1), False), (md(mp, 1), True), (md(mp, 0), False)], spin_flip_sign * Jh)
            add(H, [(md(m, 0), True), (md(m, 1), True), (md(mp, 1), False), (md(mp, 0), False)], Jh)
    # S^2 = S- S+ + Sz^2 + Sz
    Sp = np.zeros((D, D)); Sm = np.zeros((D, D))
    for m in range(5):
        add(Sp, [(md(m, 0), True), (md(m, 1), False)], 1.0)
        add(Sm, [(md(m, 1), True), (md(m, 0), False)], 1.0)
    S2 = Sm @ Sp + np.diag(Sz ** 2 + Sz)
    seniority = np.array([sum(((st >> md(m, 0)) & 1) ^ ((st >> md(m, 1)) & 1) for m in range(5)) for st in basis])
    return H, S2, Sz, seniority


def full_space_check(Delta, J, spin_flip_sign=+1.0):
    """Lowest singlet (S_z=0 sector, <S^2>=0) and lowest quintet (S_z=2) of the full ten-mode problem."""
    H, S2, Sz, sen = full_space_matrices(Delta, J, spin_flip_sign)
    comm = np.linalg.norm(H @ S2 - S2 @ H)
    sel0 = np.where(np.abs(Sz) < 1e-9)[0]
    w, v = np.linalg.eigh(H[np.ix_(sel0, sel0)])
    s2 = np.einsum("ik,ij,jk->k", v, S2[np.ix_(sel0, sel0)], v)
    singlets = np.where(np.abs(s2) < 1e-6)[0]
    e_singlet = w[singlets[0]] if len(singlets) else np.nan
    sel2 = np.where(np.abs(Sz - 2) < 1e-9)[0]
    e_quintet = np.linalg.eigvalsh(H[np.ix_(sel2, sel2)])[0]
    # lowest eigenvalue in the seniority-zero subspace of the full problem
    sel_sz = np.where(sen == 0)[0]
    e_sen0 = np.linalg.eigvalsh(H[np.ix_(sel_sz, sel_sz)])[0]
    return dict(e_singlet=float(e_singlet), e_quintet=float(e_quintet), e_sz0_lowest=float(w[0]),
                e_seniority0=float(e_sen0), commutator_H_S2=float(comm))
