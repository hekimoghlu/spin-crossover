"""Noisy density-matrix simulation, readout error, shot sampling, energy estimation and
zero-noise extrapolation (manuscript sec. 4.3, 5.1-5.2, appendix B).
"""
import numpy as np

from . import ansatz as A
from . import kanamori as K
from . import native

NQ = K.NQ
DIM = K.DIM


# ------------------------------------------------------------------ batched NumPy density-matrix kernel
def _bit(q):
    return 1 << (NQ - 1 - q)


_CNOT_PERM = {}


def _cnot_perm(c, t):
    key = (c, t)
    if key not in _CNOT_PERM:
        x = np.arange(DIM)
        _CNOT_PERM[key] = np.where(x & _bit(c), x ^ _bit(t), x)
    return _CNOT_PERM[key]


def run_batch_numpy(code, coef, params, init):
    params = np.asarray(params, float)
    N = params.shape[0]
    rho = np.zeros((N, DIM, DIM))
    rho[:, init, init] = 1.0
    for k in range(code.shape[0]):
        op, q0, q1, pi = (int(v) for v in code[k])
        if op == A.OP_RY:
            phi = coef[k] * params[:, pi]
            c, s = np.cos(phi / 2), np.sin(phi / 2)
            U = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N,2,2)
            a, b = 2 ** q0, 2 ** (NQ - 1 - q0)
            r = rho.reshape(N, a, 2, b, DIM)
            r = np.einsum("nxy,naybc->naxbc", U, r)
            r = r.reshape(N, DIM, a, 2, b)
            r = np.einsum("ncayb,nxy->ncaxb", r, U)
            rho = r.reshape(N, DIM, DIM)
        elif op == A.OP_CNOT:
            P = _cnot_perm(q0, q1)
            rho = rho[:, P][:, :, P]
        elif op == A.OP_DEP1:
            lam = params[:, pi][:, None, None, None, None, None, None]
            a, b = 2 ** q0, 2 ** (NQ - 1 - q0)
            t = rho.reshape(N, a, 2, b, a, 2, b)
            tr = t[:, :, 0, :, :, 0, :] + t[:, :, 1, :, :, 1, :]
            out = (1 - lam) * t
            lam4 = params[:, pi].reshape(N, 1, 1, 1, 1)
            for v in (0, 1):
                out[:, :, v, :, :, v, :] += 0.5 * lam4 * tr
            rho = out.reshape(N, DIM, DIM)
        elif op == A.OP_DEP2:
            p, q = sorted((q0, q1))
            a, b, c = 2 ** p, 2 ** (q - p - 1), 2 ** (NQ - 1 - q)
            t = rho.reshape(N, a, 2, b, 2, c, a, 2, b, 2, c)
            lam = params[:, pi].reshape(N, *([1] * 10))
            tr = sum(t[:, :, u, :, v, :, :, u, :, v, :] for u in (0, 1) for v in (0, 1))
            out = (1 - lam) * t
            lam8 = params[:, pi].reshape(N, *([1] * 6))
            for u in (0, 1):
                for v in (0, 1):
                    out[:, :, u, :, v, :, :, u, :, v, :] += 0.25 * lam8 * tr
            rho = out.reshape(N, DIM, DIM)
    return rho


def run_batch(code, coef, params, init, backend="auto"):
    """Final density matrices for a batch of tasks sharing one gate program."""
    if backend in ("auto", "cpp"):
        out = native.run_batch_native(NQ, code, coef, params, init)
        if out is not None:
            return out
        if backend == "cpp":
            raise RuntimeError("C++ kernel not built")
    if backend == "cuda":
        from . import cuda_backend
        return cuda_backend.run_batch(code, coef, params, init)
    return run_batch_numpy(code, coef, params, init)


# ------------------------------------------------------------------ measurement bases
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.diag([1, -1j])


def _kron_all(U):
    M = np.array([[1.0 + 0j]])
    for _ in range(NQ):
        M = np.kron(M, U)
    return M


V_X = _kron_all(_H).real
V_Y = _kron_all(_H @ _SDG)


def basis_probs(rho):
    """rho (N,32,32) real -> probabilities (N,3,32) in the Z, X, Y product bases."""
    pz = np.einsum("nii->ni", rho)
    px = np.einsum("xi,nij,xj->nx", V_X, rho, V_X)
    py = np.einsum("xi,nij,xj->nx", V_Y, rho, V_Y.conj()).real
    P = np.stack([pz, px, py], axis=1)
    P = np.clip(P, 0.0, None)
    return P / P.sum(-1, keepdims=True)


def _tensored(p, M):
    """Apply the same 2x2 matrix M (or per-row (N,2,2)) to every qubit of p (..., 32)."""
    shp = p.shape
    t = p.reshape(-1, *([2] * NQ))
    for q in range(NQ):
        t = np.moveaxis(np.tensordot(t, M, axes=([1 + q], [1])), -1, 1 + q)
    return t.reshape(shp)


def readout_matrix(e):
    return np.array([[1 - e, e], [e, 1 - e]])


def apply_readout(p, e):
    return _tensored(p, readout_matrix(e))


def correct_readout(f, e):
    return _tensored(f, np.linalg.inv(readout_matrix(e)))


def sample(rng, p, shots):
    """Multinomial histograms, p (..., 32) -> frequencies (..., 32)."""
    p = np.clip(p, 0, None)
    p = p / p.sum(-1, keepdims=True)
    counts = rng.multinomial(shots, p)
    return counts / float(shots)


# ------------------------------------------------------------------ energy from histograms
Z_MASKS = [_bit(m) for m in range(NQ)]
PAIR_MASKS = [_bit(m) | _bit(mp) for (m, mp) in K.PAIRS]


def energy_from_freqs(F, coeffs):
    """F (N,3,32) frequencies in Z,X,Y bases; coeffs = (cZ, cZZ, cXX, c0) each (N, ...)."""
    cZ, cZZ, cXX, c0 = coeffs
    N = F.shape[0]
    z = native.parity_expect(F[:, 0, :], Z_MASKS + PAIR_MASKS)
    xx = native.parity_expect(F[:, 1, :], PAIR_MASKS)
    yy = native.parity_expect(F[:, 2, :], PAIR_MASKS)
    return (c0 + np.sum(cZ * z[:, :5], 1) + np.sum(cZZ * z[:, 5:], 1)
            + np.sum(cXX * xx, 1) + np.sum(cXX * yy, 1))


def energy_exact_probs(P, coeffs):
    return energy_from_freqs(P, coeffs)


# ------------------------------------------------------------------ zero-noise extrapolation
def zne(E, scales=(1.0, 1.5, 2.0), c0=0.0):
    """E (N,3) energies at the folding scales. Exponential fit of ln|E-c0| when the three
    values of E-c0 share one sign and are monotone in s; linear fit otherwise.
    Returns (E0 (N,), used_exponential (N,) bool)."""
    s = np.asarray(scales)
    D = E - np.asarray(c0)[..., None] if np.ndim(c0) else E - c0
    same = (np.all(D > 0, 1) | np.all(D < 0, 1))
    d = np.diff(D, axis=1)
    mono = np.all(d > 0, 1) | np.all(d < 0, 1)
    use_exp = same & mono
    out = np.empty(E.shape[0])
    # linear least squares in s
    sm = s.mean()
    slope = ((s - sm) * (E - E.mean(1, keepdims=True))).sum(1) / ((s - sm) ** 2).sum()
    lin = E.mean(1) - slope * sm
    out[:] = lin
    if np.any(use_exp):
        L = np.log(np.abs(D[use_exp]))
        sl = ((s - sm) * (L - L.mean(1, keepdims=True))).sum(1) / ((s - sm) ** 2).sum()
        l0 = L.mean(1) - sl * sm
        sign = np.sign(D[use_exp][:, 0])
        cc = c0[use_exp] if np.ndim(c0) else c0
        out[use_exp] = cc + sign * np.exp(l0)
    return out, use_exp
