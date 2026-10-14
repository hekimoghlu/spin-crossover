"""ctypes bindings for the C++ (OpenMP) density-matrix kernel and the NASM parity kernel.

Both fall back to NumPy when the shared library is not built (or SCODESIGN_NO_NATIVE=1).
"""
import ctypes
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DM_PATH = os.path.join(ROOT, "cpp", "libdmsim.so")
_PAR_PATH = os.path.join(ROOT, "asm", "libparity.so")

_dm = None
_par = None


def _disabled():
    return os.environ.get("SCODESIGN_NO_NATIVE", "0") == "1"


def dm_lib():
    global _dm
    if _dm is None and not _disabled() and os.path.exists(_DM_PATH):
        lib = ctypes.CDLL(_DM_PATH)
        lib.dmsim_run_batch.restype = ctypes.c_int
        lib.dmsim_run_batch.argtypes = [
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ]
        _dm = lib
    return _dm


def par_lib():
    global _par
    if _par is None and not _disabled() and os.path.exists(_PAR_PATH):
        lib = ctypes.CDLL(_PAR_PATH)
        lib.parity_expect.restype = None
        lib.parity_expect.argtypes = [
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), ctypes.c_int64,
            np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"), ctypes.c_int64,
            ctypes.c_int64,
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ]
        lib.enum_popcount.restype = ctypes.c_int64
        lib.enum_popcount.argtypes = [ctypes.c_int64, ctypes.c_int64,
                                      np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")]
        _par = lib
    return _par


def run_batch_native(nq, code, coef, params, init):
    lib = dm_lib()
    if lib is None:
        return None
    code = np.ascontiguousarray(code, dtype=np.int32)
    coef = np.ascontiguousarray(coef, dtype=np.float64)
    params = np.ascontiguousarray(params, dtype=np.float64)
    ntask, nparam = params.shape
    D = 1 << nq
    out = np.empty((ntask, D, D), dtype=np.float64)
    rc = lib.dmsim_run_batch(nq, code.shape[0], code.ravel(), coef, ntask, nparam, params.ravel(), int(init),
                             out.reshape(-1))
    if rc != 0:
        raise RuntimeError("dmsim_run_batch failed")
    return out


def parity_expect_numpy(p, masks):
    p = np.atleast_2d(np.asarray(p, float))
    n = p.shape[1]
    x = np.arange(n, dtype=np.uint32)
    signs = np.array([1.0 - 2.0 * (np.bitwise_count(x & np.uint32(m)) & 1) for m in masks])  # (K, n)
    return p @ signs.T


def parity_expect(p, masks, force_numpy=False):
    p = np.ascontiguousarray(np.atleast_2d(np.asarray(p, float)))
    masks = np.ascontiguousarray(np.asarray(masks, dtype=np.uint32))
    lib = None if force_numpy else par_lib()
    if lib is None:
        return parity_expect_numpy(p, masks)
    out = np.empty((p.shape[0], len(masks)), dtype=np.float64)
    lib.parity_expect(p.ravel(), p.shape[1], masks, len(masks), p.shape[0], out.reshape(-1))
    return out


def enum_popcount(nbits, k, force_numpy=False):
    lib = None if force_numpy else par_lib()
    if lib is None:
        return np.array([x for x in range(1 << nbits) if bin(x).count("1") == k], dtype=np.int64)
    buf = np.empty(1 << nbits, dtype=np.int64)
    n = lib.enum_popcount(nbits, k, buf)
    return buf[:n].copy()


# ------------------------------------------------------------------ stack kernels (C++ libkernels.so, NASM)
_KER_PATH = os.path.join(ROOT, "cpp", "libkernels.so")
_ker = None
_D = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")


def ker_lib():
    global _ker
    if _ker is None and not _disabled() and os.path.exists(_KER_PATH):
        lib = ctypes.CDLL(_KER_PATH)
        lib.qaoa_expect.restype = ctypes.c_double
        lib.qaoa_expect.argtypes = [ctypes.c_int, _D, ctypes.c_int, _D, _D, ctypes.c_void_p]
        lib.qaoa_batch.restype = None
        lib.qaoa_batch.argtypes = [ctypes.c_int, _D, ctypes.c_int, ctypes.c_int, _D, _D]
        lib.rdm_forward.restype = None
        lib.rdm_forward.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _D, _D, _D, _D, _D, _D, _D,
                                    ctypes.c_double, ctypes.c_double, ctypes.c_int, _D]
        _ker = lib
    return _ker


def _c(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def qaoa_expect_native(n, f, gammas, betas, want_probs=False):
    lib = ker_lib()
    if lib is None:
        return None
    probs = np.empty(1 << n) if want_probs else None
    e = lib.qaoa_expect(n, _c(f), len(gammas), _c(gammas), _c(betas),
                        probs.ctypes.data if want_probs else None)
    return (e, probs) if want_probs else e


def qaoa_batch_native(n, f, params, p):
    lib = ker_lib()
    if lib is None:
        return None
    params = _c(params)
    out = np.empty(params.shape[0])
    lib.qaoa_batch(n, _c(f), p, params.shape[0], params.ravel(), out)
    return out


def rdm_forward_native(Xn, P, beta, T):
    lib = ker_lib()
    if lib is None:
        return None
    Xn = _c(Xn)
    N, nin = Xn.shape
    H = P["W_in"].shape[1]
    Y = np.empty((N, T))
    lib.rdm_forward(N, nin, H, Xn.ravel(), _c(P["W_in"]).ravel(), _c(P["b_in"]), _c(P["W_h"]).ravel(),
                    _c(P["W_x"]).ravel(), _c(P["b_h"]), _c(P["w_o"]), float(P["b_o"][0]), float(beta), T, Y.ravel())
    return Y


def weighted_bitsum(nbits, w, force_numpy=False):
    """out[z] = sum_q w[q] * bit_q(z), qubit q = bit (nbits-1-q).  NASM, NumPy fallback."""
    w = _c(w)
    lib = None if force_numpy else par_lib()
    if lib is None or not hasattr(lib, "weighted_bitsum"):
        z = np.arange(1 << nbits)
        bits = np.array([(z >> (nbits - 1 - q)) & 1 for q in range(nbits)], float)
        return w @ bits
    if not getattr(lib, "_wb_set", False):
        lib.weighted_bitsum.restype = None
        lib.weighted_bitsum.argtypes = [ctypes.c_int64, _D, _D]
        lib._wb_set = True
    out = np.empty(1 << nbits)
    lib.weighted_bitsum(nbits, w, out)
    return out
