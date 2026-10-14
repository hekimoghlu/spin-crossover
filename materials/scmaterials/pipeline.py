"""Compute every quantity of the materials paper that this reimplementation covers."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq

from . import design, geometry, ligandfield as lf, radiation, readout, scattering, sizemap


def _crossover_ratio(B=700.0):
    f = lambda r: lf.dH_exact(r * B, B)
    g = lambda r: lf.dH_meanfield(r * B, B)
    return brentq(f, 10, 30, xtol=1e-12), brentq(g, 10, 30, xtol=1e-12)


def thiolate_sensitivity(designs, hs=np.round(np.arange(1.4, 3.01, 0.1), 2)):
    out = {}
    for h in hs:
        D = dict(design.DONORS)
        D["thiolate"] = (design.F_SCN_S, float(h), "")
        s = design.summarise_interferents(design.interferent_map(designs, D))
        out[f"{h:.1f}"] = dict(thiolate_fp=s["fp_per_donor"]["thiolate"], n_selective_all=s["n_selective_all"])
    return out


def compute_all(verbose=True):
    t0 = time.time()
    R = {}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- ligand field + exact model over the design space ----
    ct = lf.complex_table()
    T = ct["T"]
    designs = design.design_set(ct)
    xo_exact, xo_mf = _crossover_ratio()
    R["design_space"] = dict(
        n_scaffolds=len(ct["scaffolds"]), n_complexes=int(T.size), n_pairs=len(ct["scaffolds"]) * 4,
        n_designs=len(designs),
        n_designs_by_analyte={X: sum(d["analyte"] == X for d in designs) for X in lf.ANALYTES},
        n_HS_to_LS=sum(d["direction"] == "HS->LS" for d in designs),
        Delta_min=float(ct["Delta"].min()), Delta_max=float(ct["Delta"].max()),
        B_min=float(ct["B"].min()), B_max=float(ct["B"].max()),
        max_T_resting=float(np.nanmax(T[:, 0])),
        crossover_exact_DoverB=xo_exact, crossover_mf_DoverB=xo_mf,
        tol_floor_cm=design.tol_floor_cm(),
        **design.window_stats(ct),
    )
    log(f"[{time.time()-t0:5.1f}s] design space: {len(designs)} designs")

    rows = design.interferent_map(designs)
    R["interferents"] = design.summarise_interferents(rows)
    R["interferents"]["margins_sorted"] = sorted(r["min_margin"] for r in rows)
    R["interferents"]["design_list"] = [dict(name=r["name"], analyte=r["analyte"], min_margin=r["min_margin"],
                                             fstar_imidazole=r["fstar"]["imidazole"],
                                             fp=[k for k, v in r["fp"].items() if v]) for r in rows]
    R["interferents"]["thiolate_h_sensitivity"] = thiolate_sensitivity(designs)
    log(f"[{time.time()-t0:5.1f}s] interferent map")

    R["competition"] = design.competition_budget()

    # ---- size map ----
    R["sizemap"] = dict(two_RT_kJ=2 * sizemap.R_GAS * 310 / 1e3, table=sizemap.size_table())

    # ---- geometry ----
    R["geometry"] = geometry.summary()

    # ---- readout ----
    nv = readout.nv_budget()
    nv["pitch_nm"] = {str(h): readout.critical_pitch(h) for h in (50, 100, 200)}
    R["nv"] = nv
    R["response"] = readout.response_budget()
    R["graphene"] = dict(full=readout.graphene_budget(1.0), half=readout.graphene_budget(0.5))
    log(f"[{time.time()-t0:5.1f}s] readout budgets")

    # ---- radiation ----
    R["radiation"] = radiation.budget()

    # ---- microscopy ----
    R["microscopy"] = scattering.resolvability()
    R["microscopy"]["info_limit_nm"] = scattering.INFO_LIMIT_NM
    R["microscopy"]["mfp_nm"] = 25.0
    R["runtime_s"] = time.time() - t0
    log(f"[{time.time()-t0:5.1f}s] done")
    return R
