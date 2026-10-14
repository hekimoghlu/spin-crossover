"""Simulated projected potential and kinematic electron diffraction of the
Hofmann-type cell, plus the size-dependent (Scherrer) resolvability analysis.

Electron scattering factors: first-Born (Mott-Bethe) amplitude of a Wentzel
(Yukawa-screened Coulomb) atom, f(s) = 2 Z / (a0 (4 pi^2 s^2 + 1/R^2)),
R = 0.885 a0 Z^(-1/3), s = 1/d in 1/Angstrom; isotropic Debye-Waller factor
exp(-B s^2/4) with B assumed (ASSUMPTIONS.md).  All images are SIMULATED.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import geometry
from . import kernels

A0 = 0.529177210903  # Angstrom
ZNUM = {"H": 1, "C": 6, "N": 7, "O": 8, "Fe": 26, "Pt": 78}
B_DW = 0.5           # Angstrom^2, assumed
INFO_LIMIT_NM = 0.08 # paper: information limit
SCHERRER_K = 0.9


def f_electron(Z, s):
    R = 0.885 * A0 * Z ** (-1.0 / 3.0)
    return 2.0 * Z / (A0 * (4 * np.pi ** 2 * np.asarray(s) ** 2 + 1.0 / R ** 2))


def cell_arrays(state):
    atoms, cell = geometry.hofmann_atoms(state)
    elems = sorted({e for e, _ in atoms}, key=lambda e: ZNUM[e])
    pos = np.array([p for _, p in atoms])
    sp = np.array([elems.index(e) for e, _ in atoms], dtype=np.int32)
    return pos, sp, elems, cell


def structure_factors(state, hkl, backend="auto"):
    """Kinematic structure factors F(hkl) of the tetragonal cell."""
    pos, sp, elems, cell = cell_arrays(state)
    hkl = np.asarray(hkl, float)
    g = hkl / np.array([cell["a"], cell["a"], cell["c"]])
    s = np.linalg.norm(g, axis=1)
    ftab = np.array([f_electron(ZNUM[e], s) * np.exp(-B_DW * s ** 2 / 4) for e in elems])
    return kernels.sf_sum(pos, sp, g, ftab, backend=backend), g, cell


def projected_potential(state="LS", zone="001", extent_nm=1.0, npix=128, backend="auto"):
    """Projected potential (arbitrary units, normalised to max 1) on a square
    field of view centred on an Fe column, band-limited to |g| <= 1/info limit."""
    pos, sp, elems, cell = cell_arrays(state)
    a, c = cell["a"], cell["c"]
    gmax = 1.0 / (INFO_LIMIT_NM * 10)  # 1/Angstrom
    if zone == "001":
        p1, p2 = a, a
    elif zone == "100":
        p1, p2 = a, c   # image axes (y, z)
    else:
        raise ValueError(zone)
    n1 = int(np.ceil(gmax * p1)) + 1
    n2 = int(np.ceil(gmax * p2)) + 1
    idx = [(i, j) for i in range(-n1, n1 + 1) for j in range(-n2, n2 + 1)
           if (i / p1) ** 2 + (j / p2) ** 2 <= gmax ** 2]
    if zone == "001":
        hkl = np.array([(i, j, 0) for i, j in idx], float)
    else:
        hkl = np.array([(0, i, j) for i, j in idx], float)
    F, g, _ = structure_factors(state, hkl, backend=backend)
    g2 = np.array([[i / p1, j / p2] for i, j in idx])
    L = extent_nm * 10
    x = np.linspace(-L, L, npix)
    X, Y = np.meshgrid(x, x, indexing="xy")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    V = kernels.synth_image(xy, g2, F, backend=backend).reshape(npix, npix)
    V -= V.min()
    return x / 10, V / V.max()


def reflection_list(state, qmax_nm=4.0, backend="auto"):
    """All reflections with |g| <= qmax (1/nm): returns q (1/nm), |F|^2 and hkl."""
    cell = geometry.hofmann_cell(state)
    a, c = cell["a"], cell["c"]
    qmax = qmax_nm / 10.0
    hmax = int(qmax * a) + 1
    lmax = int(qmax * c) + 1
    hkl = [(h, k, l) for h in range(-hmax, hmax + 1) for k in range(-hmax, hmax + 1)
           for l in range(-lmax, lmax + 1)
           if (h, k, l) != (0, 0, 0) and (h * h + k * k) / a ** 2 + l * l / c ** 2 <= qmax ** 2]
    F, g, _ = structure_factors(state, hkl, backend=backend)
    q = np.linalg.norm(g, axis=1) * 10
    return q, np.abs(F) ** 2, np.array(hkl)


def radial_profile(state, d_nm, q_grid, backend="auto"):
    """Powder-like radial profile: sum of |F|^2 Gaussians of FWHM = K/d (1/nm)."""
    q, I, _ = reflection_list(state, q_grid.max() + 0.5, backend=backend)
    fwhm = SCHERRER_K / d_nm
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    prof = (I[None, :] * np.exp(-0.5 * ((q_grid[:, None] - q[None, :]) / sig) ** 2)).sum(1)
    return prof


def d200(state):
    return geometry.hofmann_cell(state)["a"] / 2 / 10  # nm


def d002(state):
    return geometry.hofmann_cell(state)["c"] / 2 / 10


def resolvability():
    dls, dhs = d200("LS"), d200("HS")
    dq = 1 / dls - 1 / dhs
    shift_200 = 100 * (dhs / dls - 1)
    shift_002 = 100 * (d002("HS") / d002("LS") - 1)
    dmin = SCHERRER_K / dq
    ratio = lambda d: dq / (SCHERRER_K / d)
    surf = lambda d, delta=1.0: 100 * (1 - (1 - 2 * delta / d) ** 3)
    return dict(
        d200_LS_nm=dls, d200_HS_nm=dhs, dq_nm=dq,
        shift_inplane_pct=shift_200, shift_pillar_pct=shift_002,
        d_min_nm=dmin, ratio_20=ratio(20), ratio_50=ratio(50), ratio_5=ratio(5),
        planes_20=20 / dls, planes_50=50 / dls,
        surf_pct_20=surf(20), surf_pct_50=surf(50),
        fringe_diff_one_nm=1 / dq, fringe_diff_two_nm=2 / dq,
        fringe_diff_50=50 * dq, fringe_diff_20=20 * dq,
    )
