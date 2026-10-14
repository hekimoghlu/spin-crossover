"""CuPy RawKernel wrapper for cuda/dmsim_kernel.cu.

UNTESTED: no GPU and no nvcc/CuPy were available where this code was written, so this path
has never been compiled or executed.  `run_batch` falls back to the NumPy kernel whenever
CuPy or a CUDA device is unavailable (which is always the case in the tested environment).
"""
import os
import numpy as np

_KERNEL = None
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda", "dmsim_kernel.cu")


def available():
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        with open(_SRC) as f:
            _KERNEL = cp.RawKernel(f.read(), "dmsim_batch")
    return _KERNEL


def run_batch(code, coef, params, init, nq=5):
    if not available():
        from . import noise
        return noise.run_batch_numpy(code, coef, params, init)
    import cupy as cp
    D = 1 << nq
    params = np.ascontiguousarray(params, dtype=np.float64)
    ntask, nparam = params.shape
    d_code = cp.asarray(np.ascontiguousarray(code, dtype=np.int32).ravel())
    d_coef = cp.asarray(np.ascontiguousarray(coef, dtype=np.float64))
    d_par = cp.asarray(params.ravel())
    d_out = cp.empty(ntask * D * D, dtype=cp.float64)
    _kernel()((ntask,), (128,), (np.int32(nq), np.int32(code.shape[0]), d_code, d_coef, np.int32(nparam), d_par,
                                 np.int32(init), d_out), shared_mem=8 * D * D)
    return cp.asnumpy(d_out).reshape(ntask, D, D)


# ------------------------------------------------------------------ UNTESTED GPU counterparts of the stack kernels
_K2 = {}


def _raw(name):
    import cupy as cp
    if name not in _K2:
        with open(_SRC) as f:
            _K2[name] = cp.RawKernel(f.read(), name)
    return _K2[name]


def qaoa_batch(f, params, p):
    """Batched QAOA expectations on the GPU; falls back to the C++/NumPy path without CuPy/GPU."""
    if not available():
        from . import qaoa
        return qaoa.batch_expect(f, params, p, backend="auto")
    import cupy as cp
    f = np.ascontiguousarray(f, np.float64); params = np.ascontiguousarray(params, np.float64)
    n = int(np.log2(len(f))); nsets = params.shape[0]
    work = cp.empty(nsets * 2 * len(f), cp.float64); out = cp.empty(nsets, cp.float64)
    _raw("qaoa_batch")((nsets,), (256,), (np.int32(n), cp.asarray(f), np.int32(p), cp.asarray(params.ravel()), work, out))
    return cp.asnumpy(out)


def rdm_forward(Xn, P, beta, T):
    """Batched recurrent-depth inference on the GPU; falls back to C++ or NumPy."""
    if not available():
        from . import native
        Y = native.rdm_forward_native(Xn, P, beta, T)
        if Y is not None:
            return Y
        from .rdm import RecurrentDepth
        m = RecurrentDepth(Xn.shape[1], P["W_in"].shape[1]); m.p = P; m.beta = beta
        return np.stack(m._forward(Xn, T)[3], 1)
    import cupy as cp
    Xn = np.ascontiguousarray(Xn, np.float64); N, nin = Xn.shape; H = P["W_in"].shape[1]
    Y = cp.empty(N * T, cp.float64)
    a = lambda k: cp.asarray(np.ascontiguousarray(P[k], np.float64).ravel())
    _raw("rdm_forward")(((N + 127) // 128,), (128,), (np.int32(N), np.int32(nin), np.int32(H), cp.asarray(Xn.ravel()),
                        a("W_in"), a("b_in"), a("W_h"), a("W_x"), a("b_h"), a("w_o"), np.float64(P["b_o"][0]),
                        np.float64(beta), np.int32(T), Y))
    return cp.asnumpy(Y).reshape(N, T)
