"""Ligand-field surrogate and exact five-orbital Kanamori model of a d6 ion.

Independent reimplementation written from the method description of the
companion paper (Jorgensen rule of average environment, Kanamori
Hamiltonian with U' = 0, U = 2 J_H, J_H = P/4, P = 2.5 B + 4 C, C = 4.2 B,
seniority-zero hard-core-boson block for the low-spin state, closed-form
high-spin determinant).  All energies in cm^-1.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
G_FE2 = 10.0e3          # cm^-1, Jorgensen g(Fe2+)
B0_FE2 = 1058.0         # cm^-1, free-ion Racah B
K_FE2 = 0.24            # metal nephelauxetic factor (companion paper: assumed)
C_OVER_B = 4.2
DS_SCO = 60.0           # J mol^-1 K^-1, assumed switching entropy
T_BODY = 310.0          # K
MARGIN = 20.0           # K
CM1_TO_J_PER_MOL = 11.962656  # 1 cm^-1 = h c N_A = 11.9627 J/mol

# Table of ligand-field factors (paper table "tab:ligands")
LIGANDS = {
    # name: (f, h, role)
    "H2O": (1.00, 1.0, "scaffold"),
    "ox": (0.99, 1.5, "scaffold"),
    "gly": (1.18, 1.5, "scaffold"),
    "py": (1.23, 1.5, "scaffold"),
    "NH3": (1.25, 1.4, "scaffold"),
    "en": (1.28, 1.5, "scaffold"),
    "bipy": (1.33, 1.5, "scaffold"),
    "Cl": (0.78, 2.0, "analyte"),
    "N3": (0.83, 2.4, "analyte"),
    "NCS": (1.02, 1.6, "analyte"),
    "CN": (1.70, 2.0, "analyte"),
}
SCAFFOLD_LIGANDS = ["H2O", "ox", "gly", "py", "NH3", "en", "bipy"]
ANALYTES = ["Cl", "N3", "NCS", "CN"]
RESTING = "H2O"


def jorgensen(ligs, k=K_FE2, table=None):
    """Return (Delta, B) in cm^-1 for a list of six ligand names (or (f,h) tuples)."""
    table = LIGANDS if table is None else table
    fs, hs = [], []
    for L in ligs:
        if isinstance(L, tuple):
            f, h = L
        else:
            f, h = table[L][0], table[L][1]
        fs.append(f)
        hs.append(h)
    fbar = float(np.mean(fs))
    hbar = float(np.mean(hs))
    return G_FE2 * fbar, B0_FE2 * (1.0 - k * hbar)


def pairing_energy(B):
    return 2.5 * B + 4.0 * (C_OVER_B * B)


def kanamori_params(Delta, B):
    P = pairing_energy(B)
    J = P / 4.0
    eps = np.array([-0.4 * Delta] * 3 + [0.6 * Delta] * 2)
    return eps, J


# --------------------------------------------------------------------------
# seniority-zero (hard-core boson) block: 3 pairs in 5 orbitals, dim 10
# --------------------------------------------------------------------------
_HB_STATES = list(itertools.combinations(range(5), 3))


def hb_matrix(Delta, B):
    eps, J = kanamori_params(Delta, B)
    U = 2.0 * J
    Up = 0.0
    n = len(_HB_STATES)
    H = np.zeros((n, n))
    index = {s: i for i, s in enumerate(_HB_STATES)}
    for i, s in enumerate(_HB_STATES):
        occ = set(s)
        H[i, i] = sum(2 * eps[m] + U for m in s) + (4 * Up - 2 * J) * 3  # C(3,2)=3 pair-pairs
        for m in s:
            for mp in range(5):
                if mp in occ:
                    continue
                t = tuple(sorted((occ - {m}) | {mp}))
                H[index[t], i] += J
    return H


def e_ls_exact(Delta, B):
    return float(np.linalg.eigvalsh(hb_matrix(Delta, B))[0])


def e_hs(Delta, B):
    _, J = kanamori_params(Delta, B)
    return -0.4 * Delta - 8.0 * J


def dH_exact(Delta, B):
    """Spin-state enthalpy E_HS - E_LS (cm^-1)."""
    return e_hs(Delta, B) - e_ls_exact(Delta, B)


def dH_meanfield(Delta, B):
    _, J = kanamori_params(Delta, B)
    return 2.0 * Delta - 8.0 * J


def t_half(dH_cm, dS=DS_SCO):
    """T_1/2 = dH/dS (K); returns nan for dH <= 0 (high spin at every T)."""
    dH = np.asarray(dH_cm, dtype=float) * CM1_TO_J_PER_MOL
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dH > 0, dH / dS, np.nan)


def t_half_ligs(ligs, k=K_FE2, table=None):
    D, B = jorgensen(ligs, k=k, table=table)
    return float(t_half(dH_exact(D, B)))


# --------------------------------------------------------------------------
# full Fock-space check of the Kanamori Hamiltonian (10 spin orbitals, N=6)
# --------------------------------------------------------------------------
def _apply(ops, state):
    """Apply a product of fermion ops (rightmost first). ops: list of (mode, dagger)."""
    sign = 1
    for mode, dag in reversed(ops):
        occ = (state >> mode) & 1
        if dag and occ:
            return 0, 0
        if (not dag) and not occ:
            return 0, 0
        # parity of occupied modes below 'mode'
        below = state & ((1 << mode) - 1)
        if bin(below).count("1") % 2:
            sign = -sign
        state ^= (1 << mode)
    return sign, state


def kanamori_full(Delta, B, nelec=6, spin_flip_sign=-1.0):
    """Dense Kanamori Hamiltonian on all C(10, nelec) determinants (mode = 2m+sigma).

    spin_flip_sign multiplies J c+_{m up} c_{m dn} c+_{m' dn} c_{m' up}.  With this
    operator ordering the SU(2)-invariant (rotationally invariant) Kanamori form
    needs -1 (see ASSUMPTIONS.md); +1 is the sign as printed in the companion
    paper.  Neither sign changes the seniority-zero block or the M_S=2 determinant.
    """
    eps, J = kanamori_params(Delta, B)
    U, Up = 2.0 * J, 0.0
    basis = [s for s in range(1 << 10) if bin(s).count("1") == nelec]
    idx = {s: i for i, s in enumerate(basis)}
    nb = len(basis)
    H = np.zeros((nb, nb))
    up = lambda m: 2 * m
    dn = lambda m: 2 * m + 1

    def n(s, p):
        return (s >> p) & 1

    for i, s in enumerate(basis):
        e = 0.0
        for m in range(5):
            e += eps[m] * (n(s, up(m)) + n(s, dn(m)))
            e += U * n(s, up(m)) * n(s, dn(m))
        for m in range(5):
            for mp in range(m + 1, 5):
                nm = n(s, up(m)) + n(s, dn(m))
                nmp = n(s, up(mp)) + n(s, dn(mp))
                e += Up * nm * nmp
                e -= J * (n(s, up(m)) * n(s, up(mp)) + n(s, dn(m)) * n(s, dn(mp)))
        H[i, i] += e
        for m in range(5):
            for mp in range(5):
                if m == mp:
                    continue
                # spin flip  J c+_{m up} c_{m dn} c+_{m' dn} c_{m' up}
                sg, t = _apply([(up(m), 1), (dn(m), 0), (dn(mp), 1), (up(mp), 0)], s)
                if sg:
                    H[idx[t], i] += spin_flip_sign * J * sg
                # pair hopping J c+_{m up} c+_{m dn} c_{m' dn} c_{m' up}
                sg, t = _apply([(up(m), 1), (dn(m), 1), (dn(mp), 0), (up(mp), 0)], s)
                if sg:
                    H[idx[t], i] += J * sg
    return basis, H


def total_s2(basis):
    """S^2 matrix on the determinant basis."""
    idx = {s: i for i, s in enumerate(basis)}
    nb = len(basis)
    S2 = np.zeros((nb, nb))
    for i, s in enumerate(basis):
        nu = sum((s >> (2 * m)) & 1 for m in range(5))
        nd = sum((s >> (2 * m + 1)) & 1 for m in range(5))
        sz = 0.5 * (nu - nd)
        S2[i, i] += sz * sz - sz  # S^2 = S- S+ + Sz^2 + Sz ; use S+S- + Sz^2 - Sz
        for m in range(5):
            for mp in range(5):
                # S+ S- = sum_{m,m'} c+_{m up} c_{m dn} c+_{m' dn} c_{m' up}
                sg, t = _apply([(2 * m, 1), (2 * m + 1, 0), (2 * mp + 1, 1), (2 * mp, 0)], s)
                if sg:
                    S2[idx[t], i] += sg
    return S2


def lowest_by_spin(Delta, B, nelec=6, spin_flip_sign=-1.0):
    """Lowest energy in each total-spin sector (S=0,1,2) from the full space, S_z = 0."""
    basis, H = kanamori_full(Delta, B, nelec, spin_flip_sign)
    keep = [i for i, s in enumerate(basis)
            if sum((s >> (2 * m)) & 1 for m in range(5)) == nelec // 2]
    sub = [basis[i] for i in keep]
    Hs = H[np.ix_(keep, keep)]
    S2 = total_s2(sub)
    # S^2 commutes with H: diagonalise H + lambda S^2 is fragile; diagonalise H and read S
    w, v = np.linalg.eigh(Hs)
    s2 = np.einsum("ij,jk,ki->i", v.T, S2, v)
    out = {}
    for S in (0, 1, 2):
        sel = np.abs(s2 - S * (S + 1)) < 1e-6
        out[S] = float(w[sel].min()) if sel.any() else np.nan
    return out


# --------------------------------------------------------------------------
# enumeration of the design space
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scaffold:
    ligs: tuple  # five scaffold ligand names (sorted multiset)

    @property
    def name(self):
        counts = {}
        for L in self.ligs:
            counts[L] = counts.get(L, 0) + 1
        order = SCAFFOLD_LIGANDS
        return "+".join(f"{counts[L]}{L}" for L in order if L in counts)


def enumerate_scaffolds(max_types=3):
    out = []
    for combo in itertools.combinations_with_replacement(SCAFFOLD_LIGANDS, 5):
        if len(set(combo)) <= max_types:
            out.append(Scaffold(tuple(combo)))
    return out


def complex_table(k=K_FE2, table=None):
    """Exact T_1/2 for all scaffold x sixth-site complexes.

    Returns dict with scaffolds list, 'sixth' list and arrays Delta, B, dH, T (n_scaf x 5).
    """
    scafs = enumerate_scaffolds()
    sixth = [RESTING] + ANALYTES
    D = np.zeros((len(scafs), len(sixth)))
    Bm = np.zeros_like(D)
    dH = np.zeros_like(D)
    for i, s in enumerate(scafs):
        for j, L in enumerate(sixth):
            d, b = jorgensen(list(s.ligs) + [L], k=k, table=table)
            D[i, j], Bm[i, j] = d, b
            dH[i, j] = dH_exact(d, b)
    return dict(scaffolds=scafs, sixth=sixth, Delta=D, B=Bm, dH=dH, T=t_half(dH))


def design_ok(T_r, T_x, Tb=T_BODY, m=MARGIN):
    """Straddle condition (eq. design). nan (high spin always) is treated as T=0."""
    tr = np.nan_to_num(T_r, nan=0.0)
    tx = np.nan_to_num(T_x, nan=0.0)
    return (np.minimum(tr, tx) <= Tb - m) & (np.maximum(tr, tx) >= Tb + m)
