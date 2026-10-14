#!/usr/bin/env python3
"""Plain-script test runner (no pytest needed): python3 tests/run_tests.py"""
import math
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scmaterials import (design, geometry, kernels, ligandfield as lf, mc_ref, radiation, readout,  # noqa: E402
                         scattering, sizemap)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- model
@test
def seniority_zero_block_equals_full_space():
    rng = np.random.default_rng(12)
    for _ in range(12):
        D, B = rng.uniform(9000, 14500), rng.uniform(620, 820)
        low = lf.lowest_by_spin(D, B)
        assert abs(low[0] - lf.e_ls_exact(D, B)) < 1e-8, (low[0], lf.e_ls_exact(D, B))
        assert abs(low[2] - lf.e_hs(D, B)) < 1e-8, (low[2], lf.e_hs(D, B))


@test
def calibration_proxy_and_crossover():
    D, B = lf.jorgensen(["bipy"] * 4 + ["NCS"] * 2)
    assert round(D) == 12267 and round(B) == 669
    assert abs(lf.dH_exact(D, B) - 530) < 1
    # mean-field crossover at Delta/B = P/B = 19.3 exactly
    assert abs(lf.dH_meanfield(19.3 * 700, 700)) < 1e-9


@test
def design_space_counts():
    ct = lf.complex_table()
    assert ct["T"].size == 1505 and len(ct["scaffolds"]) == 301
    ds = design.design_set(ct)
    assert len(ds) > 0
    assert all(lf.design_ok(d["T_rest"], d["T_analyte"]) for d in ds)


# ---------------------------------------------------------------- C++ kernels
@test
def cpp_kernels_match_numpy():
    if not kernels.cpp_available():
        raise RuntimeError("C++ library not built: run make -C cpp")
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 8, (57, 3))
    sp = rng.integers(0, 4, 57).astype(np.int32)
    g = rng.normal(0, 0.6, (777, 3))
    ftab = rng.uniform(0.1, 5, (4, 777))
    a = kernels.sf_sum(pos, sp, g, ftab, backend="cpp")
    b = kernels.sf_sum(pos, sp, g, ftab, backend="numpy")
    assert np.allclose(a, b, rtol=1e-12, atol=1e-11), np.abs(a - b).max()
    xy = rng.uniform(-10, 10, (5000, 2))
    g2 = rng.normal(0, 0.5, (301, 2))
    F = rng.normal(size=301) + 1j * rng.normal(size=301)
    a = kernels.synth_image(xy, g2, F, backend="cpp")
    b = kernels.synth_image(xy, g2, F, backend="numpy")
    assert np.allclose(a, b, rtol=1e-12, atol=1e-10), np.abs(a - b).max()
    cx, cy, r = rng.uniform(0, 100, 30), rng.uniform(0, 100, 30), rng.uniform(1, 9, 30)
    a = kernels.render_spheres(200, 180, 0.5, cx, cy, r, backend="cpp")
    b = kernels.render_spheres(200, 180, 0.5, cx, cy, r, backend="numpy")
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12), np.abs(a - b).max()
    # full projected-potential pipeline through both backends
    _, V1 = scattering.projected_potential("HS", "100", npix=64, backend="cpp")
    _, V2 = scattering.projected_potential("HS", "100", npix=64, backend="numpy")
    assert np.abs(V1 - V2).max() < 1e-12


@test
def structure_factor_physics():
    # Friedel's law and F(000) = sum f(0)
    F, g, _ = scattering.structure_factors("LS", [[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    assert abs(F[0] - np.conj(F[1])) < 1e-9
    pos, sp, elems, _ = scattering.cell_arrays("LS")
    f0 = sum(scattering.f_electron(scattering.ZNUM[elems[s]], 0.0) for s in sp)
    assert abs(F[2] - f0) < 1e-9


# ---------------------------------------------------------------- Rust
def _rust_bin():
    b = mc_ref.RUST_BIN
    if not b.exists():
        raise RuntimeError("Rust binary not built: cargo build --release in rust/")
    return b


@test
def rust_mean_field_matches_python():
    b = _rust_bin()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "mf.csv"
        subprocess.run([str(b), "mf", str(out)], check=True)
        rows = mc_ref.read_csv(out)
    worst = 0.0
    for r in rows:
        g = float(r["gamma_bulk_kJmol"]) * 1e3
        if r["d_nm"] == "inf":
            ref = sizemap.loop_width(g)
        else:
            ref = sizemap.loop_width(float(sizemap.gamma_of_d(float(r["d_nm"]), g)))
        worst = max(worst, abs(ref - float(r["loop_width_K"])))
        assert abs(float(r["d_c_nm"]) - sizemap.d_critical(g)) < 1e-9
    assert worst < 1e-6, worst


@test
def rust_monte_carlo_matches_python_bit_for_bit():
    _rust_bin()
    with tempfile.TemporaryDirectory() as td:
        for d, g, seed in ((6.0, 10000.0, 42), (5.0, 12000.0, 7)):
            n, ts, up, dn = mc_ref.run_rust_loop(Path(td) / "l.csv", d, g, nt=9, sweeps=15, seed=seed)
            nb = mc_ref.build_particle(d)
            ts2, up2, dn2 = mc_ref.hysteresis(nb, g, n_t=9, sweeps=15, seed=seed)
            assert n == len(nb)
            assert up == up2 and dn == dn2, (up, up2)


# ---------------------------------------------------------------- closed forms
@test
def sizemap_dc_is_loop_closure():
    for g in (6e3, 8e3, 10e3):
        dc = sizemap.d_critical(g)
        assert sizemap.loop_width(float(sizemap.gamma_of_d(dc * 0.999, g))) == 0.0
        assert sizemap.loop_width(float(sizemap.gamma_of_d(dc * 1.05, g))) > 0.0


@test
def sizemap_sweep_consistent_with_turning_points():
    g = 9e3
    Ts, xh, xc = sizemap.mf_hysteresis_sweep(g, Tmin=250, Tmax=370, nT=1201)
    up = Ts[np.argmax(xh > 0.5)]
    dn = Ts[len(xc) - 1 - np.argmax(xc[::-1] < 0.5)]
    assert abs((up - dn) - sizemap.loop_width(g)) < 0.25


@test
def dipole_and_pitch_scale_invariance():
    b1 = readout.stray_field_contrast(50, 100)
    b2 = readout.stray_field_contrast(100, 200)
    assert abs(b1 / b2 - 1.0) < 1e-12  # m ~ d^3, B ~ m/r^3
    assert abs(readout.critical_pitch(100) / readout.critical_pitch(50) - 2) < 1e-3


@test
def bethe_water_close_to_pstar():
    S = radiation.bethe_mass_stopping(10.0, radiation.WATER, I_override=75.0)
    assert abs(S / 45.67 - 1) < 0.01  # NIST PSTAR 10 MeV protons in water


@test
def geometry_rigid_linkage():
    c = geometry.hofmann_cell("LS")
    assert abs(c["a"] - math.sqrt(2) * (1.97 + geometry.R_CN + geometry.R_PTC)) < 1e-12
    atoms, cell = geometry.hofmann_atoms("HS")
    fe = [p for e, p in atoms if e == "Fe"][0]
    n_pz = [p for e, p in atoms if e == "N" and p[2] > 0.1]
    assert abs(min(np.linalg.norm(p - fe) for p in n_pz) - 2.17) < 1e-9


@test
def cuda_module_is_not_used_by_default():
    # The CUDA path is untested; it must never be chosen implicitly.
    assert kernels._choose("auto") in ("cpp", "numpy")


def main():
    t0 = time.time()
    fails = 0
    for fn in TESTS:
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time()-t:.1f}s)")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS)-fails}/{len(TESTS)} passed in {time.time()-t0:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
