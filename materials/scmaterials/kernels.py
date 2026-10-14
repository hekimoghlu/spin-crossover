"""Compute kernels with three backends: C++/OpenMP (ctypes), CUDA (CuPy, untested), NumPy.

backend selection: argument `backend` in {"auto", "cpp", "numpy", "cuda"}; "auto"
uses the C++ library if it has been built (cpp/build/libscmk.so) and NumPy
otherwise.  CUDA is only used when explicitly requested (it has never been
compiled or run in the environment this code was written in).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_LIB = None
_LIB_PATH = Path(__file__).resolve().parent.parent / "cpp" / "build" / "libscmk.so"


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = Path(os.environ.get("SCMK_LIB", _LIB_PATH))
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    d = ctypes.POINTER(ctypes.c_double)
    i32 = ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.c_int64
    lib.sf_sum.argtypes = [i64, d, i32, i64, d, d, d, d]
    lib.sf_sum.restype = None
    lib.synth_image.argtypes = [i64, d, i64, d, d, d, d]
    lib.synth_image.restype = None
    lib.render_spheres.argtypes = [i64, i64, ctypes.c_double, i64, d, d, d, d]
    lib.render_spheres.restype = None
    lib.scmk_num_threads.restype = ctypes.c_int
    _LIB = lib
    return lib


def cpp_available():
    return _load() is not None


def _ptr(a, ct=ctypes.c_double):
    return a.ctypes.data_as(ctypes.POINTER(ct))


def _choose(backend):
    if backend == "auto":
        return "cpp" if cpp_available() else "numpy"
    if backend == "cpp" and not cpp_available():
        raise RuntimeError("C++ kernel library not built (run make -C cpp)")
    return backend


# --------------------------------------------------------------------------
def sf_sum(pos, species, g, ftab, backend="auto"):
    """F(g) = sum_j ftab[species_j, g] exp(2 pi i g.r_j). pos (N,3), g (G,3), ftab (S,G)."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    species = np.ascontiguousarray(species, dtype=np.int32)
    g = np.ascontiguousarray(g, dtype=np.float64)
    ftab = np.ascontiguousarray(ftab, dtype=np.float64)
    b = _choose(backend)
    if b == "cpp":
        lib = _load()
        re = np.empty(len(g))
        im = np.empty(len(g))
        lib.sf_sum(len(pos), _ptr(pos), _ptr(species, ctypes.c_int32), len(g), _ptr(g),
                   _ptr(ftab), _ptr(re), _ptr(im))
        return re + 1j * im
    if b == "cuda":
        from . import cuda_kernels
        return cuda_kernels.sf_sum(pos, species, g, ftab)
    out = np.empty(len(g), dtype=complex)
    for s in range(0, len(g), 4096):  # chunk to bound memory
        gg = g[s:s + 4096]
        ph = 2 * np.pi * (gg @ pos.T)                    # (G, N)
        f = ftab[:, s:s + 4096][species, :].T            # (G, N)
        out[s:s + 4096] = (f * np.exp(1j * ph)).sum(axis=1)
    return out


def synth_image(xy, g2, F, backend="auto"):
    """V(r) = Re sum_g F(g) exp(2 pi i g.r) at points xy (P,2) for 2D g-vectors g2 (G,2)."""
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    g2 = np.ascontiguousarray(g2, dtype=np.float64)
    Fr = np.ascontiguousarray(np.real(F), dtype=np.float64)
    Fi = np.ascontiguousarray(np.imag(F), dtype=np.float64)
    b = _choose(backend)
    if b == "cpp":
        lib = _load()
        out = np.empty(len(xy))
        lib.synth_image(len(xy), _ptr(xy), len(g2), _ptr(g2), _ptr(Fr), _ptr(Fi), _ptr(out))
        return out
    if b == "cuda":
        from . import cuda_kernels
        return cuda_kernels.synth_image(xy, g2, Fr + 1j * Fi)
    out = np.empty(len(xy))
    for s in range(0, len(xy), 8192):
        ph = 2 * np.pi * (xy[s:s + 8192] @ g2.T)
        out[s:s + 8192] = (np.cos(ph) * Fr - np.sin(ph) * Fi).sum(axis=1)
    return out


def render_spheres(nx, ny, px, cx, cy, rad, backend="auto"):
    """Projected thickness map (ny, nx) of spheres; pixel centres at (i+0.5)*px."""
    cx = np.ascontiguousarray(cx, dtype=np.float64)
    cy = np.ascontiguousarray(cy, dtype=np.float64)
    rad = np.ascontiguousarray(rad, dtype=np.float64)
    b = _choose(backend)
    if b == "cpp":
        lib = _load()
        out = np.empty((ny, nx))
        lib.render_spheres(nx, ny, float(px), len(cx), _ptr(cx), _ptr(cy), _ptr(rad), _ptr(out))
        return out
    if b == "cuda":
        raise NotImplementedError("render_spheres has no CUDA kernel")
    x = (np.arange(nx) + 0.5) * px
    y = (np.arange(ny) + 0.5) * px
    out = np.zeros((ny, nx))
    for xi, yi, ri in zip(cx, cy, rad):
        # bounding box for efficiency
        i0, i1 = max(0, int((yi - ri) / px) - 1), min(ny, int((yi + ri) / px) + 2)
        j0, j1 = max(0, int((xi - ri) / px) - 1), min(nx, int((xi + ri) / px) + 2)
        if i0 >= i1 or j0 >= j1:
            continue
        dx = x[j0:j1][None, :] - xi
        dy = y[i0:i1][:, None] - yi
        r2 = ri * ri - dx * dx - dy * dy
        out[i0:i1, j0:j1] += np.where(r2 > 0, 2 * np.sqrt(np.clip(r2, 0, None)), 0.0)
    return out
