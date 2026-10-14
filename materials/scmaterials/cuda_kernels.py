"""CUDA versions of the structure-factor sum and the image synthesis (CuPy RawKernel).

*** UNTESTED. ***  This module has never been compiled or executed: the
environment in which this reimplementation was written has no GPU, no nvcc
and no CuPy.  It is provided as a starting point only.  Nothing in run_all.sh,
the tests or the reported results uses it; scmaterials.kernels only imports it
when backend="cuda" is requested explicitly, and falls back to NumPy/C++
otherwise.  The kernel source is also in cuda/scmk_kernels.cu.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_SRC = (Path(__file__).resolve().parent.parent / "cuda" / "scmk_kernels.cu")


def _module():
    import cupy as cp  # noqa: F401  (raises ImportError where CuPy is absent)
    src = _SRC.read_text()
    return cp.RawModule(code=src, options=("-std=c++14",))


def sf_sum(pos, species, g, ftab, block=128):
    import cupy as cp
    mod = _module()
    k = mod.get_function("sf_sum_kernel")
    d_pos = cp.asarray(pos, dtype=cp.float64)
    d_sp = cp.asarray(species, dtype=cp.int32)
    d_g = cp.asarray(g, dtype=cp.float64)
    d_f = cp.asarray(ftab, dtype=cp.float64)
    n_g = len(g)
    re = cp.empty(n_g, dtype=cp.float64)
    im = cp.empty(n_g, dtype=cp.float64)
    grid = ((n_g + block - 1) // block,)
    k(grid, (block,), (np.int64(len(pos)), d_pos, d_sp, np.int64(n_g), d_g, d_f, re, im))
    return cp.asnumpy(re) + 1j * cp.asnumpy(im)


def synth_image(xy, g2, F, block=128):
    import cupy as cp
    mod = _module()
    k = mod.get_function("synth_image_kernel")
    d_xy = cp.asarray(xy, dtype=cp.float64)
    d_g = cp.asarray(g2, dtype=cp.float64)
    d_re = cp.asarray(np.real(F), dtype=cp.float64)
    d_im = cp.asarray(np.imag(F), dtype=cp.float64)
    n = len(xy)
    out = cp.empty(n, dtype=cp.float64)
    grid = ((n + block - 1) // block,)
    k(grid, (block,), (np.int64(n), d_xy, np.int64(len(g2)), d_g, d_re, d_im, out))
    return cp.asnumpy(out)
