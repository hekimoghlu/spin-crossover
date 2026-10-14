"""Design set, interferent (selectivity) map and window statistics.

Takes the exact five-orbital switching temperatures of the full ligand space
(ligandfield.complex_table) and re-evaluates every successful design with
physiological donors in the labile sixth site.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import ligandfield as lf

# Physiological donors and their ligand-field factors (f, h).  Proxies follow
# the paper (pyridine for imidazole, ammonia for amine, oxalate for
# carboxylate, S-bound thiocyanate for thiolate).  Hydroxide/phosphate and the
# h of S-bound thiocyanate are not given in the paper: see ASSUMPTIONS.md.
F_SCN_S = 0.73          # Jorgensen f for S-bound thiocyanate (tabulated value)
H_SCN_S_ASSUMED = 2.0   # assumed: monoanionic soft-donor class (as Cl-, CN- in table)
DONORS = {
    "hydroxide": (0.99, 1.5, "assumed: O-donor anion class, oxalate factors"),
    "phosphate": (0.99, 1.5, "assumed: O-donor anion class, oxalate factors"),
    "carboxylate": (0.99, 1.5, "proxy: oxalate"),
    "imidazole": (1.23, 1.5, "proxy: pyridine"),
    "amine": (1.25, 1.4, "proxy: ammonia"),
    "thiolate": (F_SCN_S, H_SCN_S_ASSUMED, "proxy: S-bound thiocyanate (h assumed)"),
}


def design_set(ct=None, Tb=lf.T_BODY, m=lf.MARGIN):
    ct = lf.complex_table() if ct is None else ct
    T = ct["T"]
    designs = []
    for i, s in enumerate(ct["scaffolds"]):
        for j, X in enumerate(lf.ANALYTES, start=1):
            if lf.design_ok(T[i, 0], T[i, j], Tb, m):
                designs.append(dict(scaffold=s, name=s.name, analyte=X,
                                    T_rest=float(np.nan_to_num(T[i, 0])),
                                    T_analyte=float(np.nan_to_num(T[i, j])),
                                    direction="HS->LS" if np.nan_to_num(T[i, j]) > np.nan_to_num(T[i, 0]) else "LS->HS"))
    return designs


def _T(ligs, k=lf.K_FE2):
    return float(np.nan_to_num(lf.t_half_ligs(ligs, k=k), nan=0.0))


def critical_f(scaffold, h_sixth, Tb=lf.T_BODY, k=lf.K_FE2):
    """Sixth-site spectrochemical factor f* at which the complex crosses Tb."""
    def g(f):
        return _T(list(scaffold.ligs) + [(f, h_sixth)], k=k) - Tb
    lo, hi = 0.0, 4.0
    if g(lo) > 0 or g(hi) < 0:
        return np.nan
    # T is 0 (nan) for a range of small f: find a bracket with a sign change
    return brentq(g, lo, hi, xtol=1e-10)


def on_analyte_side(T_donor, T_analyte, Tb=lf.T_BODY, m=lf.MARGIN):
    """True if the donor complex sits where the analyte complex sits (false positive)."""
    if T_analyte >= Tb + m:
        return T_donor >= Tb + m
    return T_donor <= Tb - m


def interferent_map(designs, donors=None, Tb=lf.T_BODY, m=lf.MARGIN, k=lf.K_FE2):
    donors = DONORS if donors is None else donors
    rows = []
    for d in designs:
        s = d["scaffold"]
        up = d["T_analyte"] > d["T_rest"]  # analyte drives to low spin (higher T1/2)
        row = dict(name=d["name"], analyte=d["analyte"], fp={}, margin={}, fstar={}, T_donor={})
        for dn, (f, h, _) in donors.items():
            Td = _T(list(s.ligs) + [(f, h)], k=k)
            fs = critical_f(s, h, Tb, k)
            row["T_donor"][dn] = Td
            row["fstar"][dn] = fs
            row["margin"][dn] = (fs - f) if up else (f - fs)
            row["fp"][dn] = bool(on_analyte_side(Td, d["T_analyte"], Tb, m))
        # cross-analyte silence
        row["cross_fp"] = {}
        for X in lf.ANALYTES:
            if X == d["analyte"]:
                continue
            Tx = _T(list(s.ligs) + [X], k=k)
            row["cross_fp"][X] = bool(on_analyte_side(Tx, d["T_analyte"], Tb, m))
        row["min_margin_donor"] = min(row["margin"], key=row["margin"].get)
        row["min_margin"] = row["margin"][row["min_margin_donor"]]
        rows.append(row)
    return rows


def summarise_interferents(rows):
    margins = np.array([r["min_margin"] for r in rows])
    fp_any = np.array([any(r["fp"].values()) for r in rows])
    cross_any = np.array([any(r["cross_fp"].values()) for r in rows])
    imid = [r for r in rows if r["min_margin_donor"] == "imidazole"]
    fstar_imid = np.array([r["fstar"]["imidazole"] for r in rows])
    fp_scaffolds = sorted({r["name"] for r, f in zip(rows, fp_any) if f})
    per_donor = {dn: int(sum(r["fp"][dn] for r in rows)) for dn in rows[0]["fp"]}
    return dict(
        n_designs=len(rows),
        n_selective_all=int((~fp_any).sum()),
        n_fp_designs=int(fp_any.sum()),
        fp_scaffolds=fp_scaffolds,
        fp_per_donor=per_donor,
        n_cross_silent=int((~cross_any).sum()),
        cross_fp_designs=sorted({f"{r['name']}|{r['analyte']}" for r, c in zip(rows, cross_any) if c}),
        min_margin_is_imidazole_all=len(imid) == len(rows),
        min_margin_donor_counts={dn: int(sum(r["min_margin_donor"] == dn for r in rows)) for dn in rows[0]["fp"]},
        margin_median=float(np.median(margins)),
        margin_min=float(margins.min()),
        margin_max=float(margins.max()),
        n_margin_pos=int((margins > 0).sum()),
        n_margin_gt_0p1=int((margins > 0.1).sum()),
        n_margin_gt_0p2=int((margins > 0.2).sum()),
        n_margin_nonpos=int((margins <= 0).sum()),
        fstar_imidazole_min=float(np.nanmin(fstar_imid)),
        fstar_imidazole_median=float(np.nanmedian(fstar_imid)),
        fstar_imidazole_max=float(np.nanmax(fstar_imid)),
    )


def window_stats(ct, lo=300.0, hi=325.0, target=312.0):
    T = ct["T"].ravel()
    T = T[np.isfinite(T)]
    w = (T >= lo) & (T <= hi)
    return dict(n_window=int(w.sum()), nearest_below=float(T[T < target].max()),
                nearest_above=float(T[T > target].min()))


# --------------------------------------------------------------------------
# competition budget
# --------------------------------------------------------------------------
PLASMA = [  # (donor, concentration in M)  -- nominal values from the paper's table
    ("chloride", 0.1), ("bicarbonate", 25e-3), ("phosphate", 1e-3),
    ("albumin", 0.6e-3), ("histidine", 80e-6), ("cysteine", 30e-6),
    ("thiocyanate", 50e-6), ("glutathione (intracellular)", 2e-3),
]
DETECTION = {"cyanide": 20e-6, "azide": 20e-6, "thiocyanate": 50e-6, "chloride": 0.1}


def competition_budget():
    """Required K_X/K_i >= c_i/c_X for half occupation by X against donor i."""
    return {X: {dn: c / cX for dn, c in PLASMA} for X, cX in DETECTION.items()}


def tol_floor_cm(dT=10.0, dS=lf.DS_SCO):
    """Spin-state enthalpy (cm^-1) equivalent to dT kelvin of switching temperature."""
    return dT * dS / lf.CM1_TO_J_PER_MOL
