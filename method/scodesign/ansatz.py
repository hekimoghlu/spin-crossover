"""Givens-rotation ansatz, noiseless angle optimisation, gate compilation, routing, folding
(manuscript sec. 4.2 and appendix A).

Gate convention (documented in ASSUMPTIONS.md): Ry(phi) = [[cos phi/2, -sin phi/2], [sin phi/2, cos phi/2]].
With it, the compiled G_ia(theta) = CNOT_{a->i} CRy(2 theta)_{i->a} CNOT_{a->i} maps
|1_i 0_a> -> cos(theta)|10> + sin(theta)|01>; the manuscript writes -sin(theta), i.e. the
opposite sign of theta, which the optimiser absorbs.
"""
import numpy as np
from scipy.optimize import minimize

from . import kanamori as K

NQ = K.NQ
DIM = K.DIM
GIVENS_ORDER = [(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)]
# linear chain 0-3-1-4-2: position of each logical qubit on the chain
CHAIN = [0, 3, 1, 4, 2]
LAYOUT = {q: CHAIN.index(q) for q in range(NQ)}  # logical -> physical position

# op codes (shared with the C++ / CUDA kernels)
OP_RY, OP_CNOT, OP_DEP1, OP_DEP2 = 0, 1, 2, 3
# parameter slots: 0..5 Givens angles, 6 = lambda_1q, 7 = lambda_2q
P_L1, P_L2 = 6, 7
NPARAM = 8


# ------------------------------------------------------------------ direct statevector ansatz
def _givens_index_pairs():
    out = []
    for (i, a) in GIVENS_ORDER:
        src, dst = [], []
        for x in range(DIM):
            if K.bit(x, i) == 1 and K.bit(x, a) == 0:
                y = x ^ (1 << (NQ - 1 - i)) ^ (1 << (NQ - 1 - a))
                src.append(x); dst.append(y)
        out.append((np.array(src), np.array(dst)))
    return out


_GIDX = _givens_index_pairs()


def ansatz_state(thetas):
    psi = np.zeros(DIM); psi[K.REF_INDEX] = 1.0
    for th, (src, dst) in zip(thetas, _GIDX):
        c, s = np.cos(th), np.sin(th)
        x, y = psi[src].copy(), psi[dst].copy()
        psi[src] = c * x - s * y
        psi[dst] = s * x + c * y
    return psi


def optimise_angles(Hmat, n_starts=4, seed=0):
    """Noiseless minimisation of <psi(theta)|H|psi(theta)> over the six angles."""
    Hr = np.real(Hmat)
    f = lambda t: float(ansatz_state(t) @ Hr @ ansatz_state(t))
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.zeros(6)] + [rng.normal(0, 0.3, 6) for _ in range(n_starts - 1)]
    for x0 in starts:
        r = minimize(f, x0, method="BFGS", options=dict(gtol=1e-10, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


# ------------------------------------------------------------------ compilation to gate list
def givens_ops(i, a, pidx):
    """Appendix A: G_ia = CNOT_{a->i} . CRy(2θ)_{i->a} . CNOT_{a->i};
    CRy(φ) = Ry(φ/2)_a CNOT_{i->a} Ry(-φ/2)_a CNOT_{i->a}  (time order: right to left).
    Returns ops in time order: (code, q0, q1, param_index, coef)."""
    return [
        (OP_CNOT, a, i, -1, 0.0),
        (OP_CNOT, i, a, -1, 0.0),
        (OP_RY, a, -1, pidx, -1.0),   # Ry(-φ/2) with φ = 2θ  ->  Ry(-θ)
        (OP_CNOT, i, a, -1, 0.0),
        (OP_RY, a, -1, pidx, +1.0),   # Ry(+φ/2) = Ry(+θ)
        (OP_CNOT, a, i, -1, 0.0),
    ]




def swap_ops(p, q):
    return [(OP_CNOT, p, q, -1, 0.0), (OP_CNOT, q, p, -1, 0.0), (OP_CNOT, p, q, -1, 0.0)]


def logical_gate_list(routed=False):
    """Gate list (no noise) on wires.  For routed circuits wires are chain positions."""
    ops = []
    if not routed:
        for k, (i, a) in enumerate(GIVENS_ORDER):
            ops += givens_ops(i, a, k)
        return ops
    pos = dict(LAYOUT)            # logical -> position
    occ = {v: k for k, v in pos.items()}  # position -> logical
    for k, (i, a) in enumerate(GIVENS_ORDER):
        pi, pa = pos[i], pos[a]
        swaps = []
        if abs(pi - pa) > 1:
            # move occupied qubit i next to virtual a, one SWAP at a time
            step = 1 if pa > pi else -1
            while abs(pos[i] - pos[a]) > 1:
                p0 = pos[i]; p1 = p0 + step
                swaps.append((p0, p1))
                ops += swap_ops(p0, p1)
                li, lo = occ[p0], occ[p1]
                pos[li], pos[lo] = p1, p0
                occ[p0], occ[p1] = lo, li
        ops += givens_ops(pos[i], pos[a], k)
        for (p0, p1) in reversed(swaps):
            ops += swap_ops(p0, p1)
            l0, l1 = occ[p0], occ[p1]
            pos[l0], pos[l1] = p1, p0
            occ[p0], occ[p1] = l1, l0
    assert pos == LAYOUT
    return ops


def initial_index(routed=False):
    """Computational-basis index of |11100> on the wires."""
    x = 0
    for q in (0, 1, 2):
        w = LAYOUT[q] if routed else q
        x |= 1 << (NQ - 1 - w)
    return x


def wire_to_logical_perm(routed=False):
    """perm[x_wire] = x_logical for relabelling the final density matrix."""
    perm = np.zeros(DIM, dtype=int)
    for xw in range(DIM):
        xl = 0
        for q in range(NQ):
            w = LAYOUT[q] if routed else q
            if (xw >> (NQ - 1 - w)) & 1:
                xl |= 1 << (NQ - 1 - q)
        perm[xw] = xl
    return perm


def fold_selection(n_cnot, seed=12345):
    """Fixed random order of CNOT indices; scale s folds the first round((s-1)/2 * n) of them."""
    return np.random.default_rng(seed).permutation(n_cnot)


def noisy_program(routed=False, scale=1.0, fold_seed=12345):
    """Gate list with folding and depolarising ops inserted after every gate.

    Returns an int array (nops, 4) [code, q0, q1, pidx] and a float array coef (nops,).
    For OP_RY: angle = coef * params[pidx]; for OP_DEP1/2: lambda = params[pidx].
    """
    base = logical_gate_list(routed)
    cnot_pos = [k for k, op in enumerate(base) if op[0] == OP_CNOT]
    order = fold_selection(len(cnot_pos), fold_seed)
    nfold = int(round((scale - 1.0) / 2.0 * len(cnot_pos)))
    folded = set(cnot_pos[j] for j in order[:nfold])
    ops = []
    for k, op in enumerate(base):
        reps = 3 if k in folded else 1
        for _ in range(reps):
            ops.append(op)
            if op[0] == OP_CNOT:
                ops.append((OP_DEP2, op[1], op[2], P_L2, 0.0))
            else:
                ops.append((OP_DEP1, op[1], -1, P_L1, 0.0))
    code = np.array([[o[0], o[1], o[2], o[3]] for o in ops], dtype=np.int32)
    coef = np.array([o[4] for o in ops], dtype=np.float64)
    return code, coef


def gate_counts(routed=False, scale=1.0):
    code, _ = noisy_program(routed, scale)
    return int(np.sum(code[:, 0] == OP_CNOT)), int(np.sum(code[:, 0] == OP_RY))


# ------------------------------------------------------------------ noiseless circuit simulator (for tests/QASM)
def ry(phi):
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, -s], [s, c]])


def apply_1q(psi, U, q):
    t = psi.reshape([2] * NQ)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def apply_cnot(psi, c, t):
    out = psi.copy()
    for x in range(DIM):
        if (x >> (NQ - 1 - c)) & 1:
            y = x ^ (1 << (NQ - 1 - t))
            out[y] = psi[x]
    return out


def run_statevector(ops, thetas, init):
    psi = np.zeros(DIM, dtype=complex); psi[init] = 1.0
    for op in ops:
        if op[0] == OP_RY:
            psi = apply_1q(psi, ry(op[4] * thetas[op[3]]), op[1])
        elif op[0] == OP_CNOT:
            psi = apply_cnot(psi, op[1], op[2])
    return psi
