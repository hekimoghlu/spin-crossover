"""SIMULATED bright-field micrographs of particle ensembles and a lattice image
of one particle.  Mass-thickness contrast I/I0 = exp(-t/Lambda) (paper), with
Lambda = 25 nm; size distribution, dose, noise and blur are assumptions
(ASSUMPTIONS.md)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import kernels, scattering

MFP_NM = 25.0          # paper: assumed effective mean free path
SIGMA_LN = 0.25        # assumed lognormal width of the diameter distribution
DOSE_E_PER_PX = 400.0  # assumed electrons per pixel (shot noise)
PSF_PX = 1.0           # assumed Gaussian blur (pixels)


def sample_ensemble(mean_d_nm, field_nm, n_target, seed, sigma_ln=SIGMA_LN, gap_nm=0.5, max_tries=200000):
    """Random sequential addition of non-overlapping (in projection) particles
    with lognormal diameters of arithmetic mean mean_d_nm."""
    rng = np.random.default_rng(seed)
    mu = np.log(mean_d_nm) - 0.5 * sigma_ln ** 2
    cx, cy, r = [], [], []
    tries = 0
    while len(r) < n_target and tries < max_tries:
        tries += 1
        d = rng.lognormal(mu, sigma_ln)
        R = d / 2
        x, y = rng.uniform(R, field_nm - R, size=2)
        if r:
            dd = np.hypot(np.array(cx) - x, np.array(cy) - y)
            if np.any(dd < np.array(r) + R + gap_nm):
                continue
        cx.append(x); cy.append(y); r.append(R)
    return np.array(cx), np.array(cy), np.array(r)


def bright_field(mean_d_nm, field_nm, npix=256, n_target=40, seed=0, backend="auto"):
    cx, cy, r = sample_ensemble(mean_d_nm, field_nm, n_target, seed)
    px = field_nm / npix
    t = kernels.render_spheres(npix, npix, px, cx, cy, r, backend=backend)
    I = np.exp(-t / MFP_NM)
    I = gaussian_filter(I, PSF_PX)
    rng = np.random.default_rng(seed + 1000)
    I = rng.poisson(I * DOSE_E_PER_PX) / DOSE_E_PER_PX
    return I, 2 * r, px


def lattice_particle(d_nm=12.0, field_nm=16.0, npix=320, state="LS", seed=0, backend="auto"):
    """[001] lattice image of one particle: band-limited projected potential of
    the periodic cell weighted by the sphere's projected thickness (phase-object
    contrast, weak-phase approximation), plus noise.  Returns image and FFT."""
    x, V = None, None
    # projected potential over the whole field, computed directly at the pixel grid
    pos, sp, elems, cell = scattering.cell_arrays(state)
    a = cell["a"]
    gmax = 1.0 / (scattering.INFO_LIMIT_NM * 10)
    n1 = int(np.ceil(gmax * a)) + 1
    idx = [(i, j) for i in range(-n1, n1 + 1) for j in range(-n1, n1 + 1)
           if (i * i + j * j) / a ** 2 <= gmax ** 2]
    hkl = np.array([(i, j, 0) for i, j in idx], float)
    F, g, _ = scattering.structure_factors(state, hkl, backend=backend)
    g2 = np.array([[i / a, j / a] for i, j in idx])
    L = field_nm * 10
    xs = (np.arange(npix) + 0.5) * L / npix
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    Vp = kernels.synth_image(np.column_stack([X.ravel(), Y.ravel()]), g2, F, backend=backend)
    Vp = Vp.reshape(npix, npix)
    Vp = (Vp - Vp.min()) / (Vp.max() - Vp.min())
    t = kernels.render_spheres(npix, npix, field_nm / npix, np.array([field_nm / 2]),
                               np.array([field_nm / 2]), np.array([d_nm / 2]), backend=backend)
    img = 1.0 - 0.6 * Vp * (t / d_nm) - 0.15 * (t / d_nm)
    rng = np.random.default_rng(seed)
    img = rng.poisson(np.clip(img, 0, None) * DOSE_E_PER_PX) / DOSE_E_PER_PX
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))))
    return img, fft, field_nm / npix
