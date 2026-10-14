#!/usr/bin/env python3
"""Compare results/results.json with the numbers reported in the manuscript.

Paper values are READ from the manuscript's \\newcommand block and table bodies
only for this comparison; nothing here feeds back into any computed result.
Writes VALIDATION.md.

Verdicts: "match" = our value, rounded to the precision printed in the paper,
equals the paper value; "near" = differs by at most one unit in the last
printed digit or 1 %; "MISMATCH" otherwise.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = ROOT.parent / "papers" / "paperB.tex"


def parse_macros(tex):
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", tex, re.M):
        out[m.group(1)] = m.group(2)
    return out


def to_number(s):
    s = s.replace("~per cent", "").replace(",", "").replace("{", "").replace("}", "").strip()
    m = re.match(r"^([+-]?[0-9.]+)\$\\times\s*10\^\-?([0-9-]+)\$$", s.replace(" ", ""))
    if m:
        exp = re.search(r"10\^\{?(-?[0-9]+)", s)
        return float(m.group(1)) * 10 ** int(exp.group(1)), m.group(1)
    try:
        return float(s), s
    except ValueError:
        return None, s


def decimals(txt):
    txt = txt.lstrip("+-")
    return len(txt.split(".")[1]) if "." in txt else 0


def verdict(ours, paper_val, paper_txt):
    if ours is None or paper_val is None:
        return "n/a"
    dec = decimals(paper_txt)
    sig_scale = 1.0
    if paper_val != 0 and ("e" in f"{paper_val}" or abs(paper_val) < 1e-2):
        # scientific: compare mantissas
        import math
        e = math.floor(math.log10(abs(paper_val)))
        sig_scale = 10 ** e
        if round(ours / sig_scale, dec) == round(paper_val / sig_scale, dec):
            return "match"
        unit = 10 ** -dec * sig_scale
    else:
        if round(ours, dec) == round(paper_val, dec) or f"{ours:.{dec}f}" == f"{paper_val:.{dec}f}":
            return "match"
        unit = 10 ** -dec
    if abs(ours - paper_val) <= unit + 1e-12 or abs(ours - paper_val) <= 0.01 * abs(paper_val):
        return "near"
    return "MISMATCH"


def get(R, path):
    cur = R
    for p in path.split("/"):
        cur = cur[p]
    return cur


MAP = [
    # macro, json path, transform, description
    ("nComplexes", "design_space/n_complexes", None, "complexes enumerated"),
    ("nPairs", "design_space/n_pairs", None, "scaffold-analyte pairs"),
    ("nbDesigns", "design_space/n_designs", None, "designs passing straddle condition (m=20 K)"),
    ("tolFloorCm", "design_space/tol_floor_cm", None, "10 K of T1/2 in cm^-1"),
    ("nWindowThreeHundred", "design_space/n_window", None, "complexes with 300<=T1/2<=325 K"),
    ("nearestBelowFever", "design_space/nearest_below", None, "nearest T1/2 below 312 K"),
    ("nearestAboveFever", "design_space/nearest_above", None, "nearest T1/2 above 312 K"),
    ("nbSelectiveAll", "interferents/n_selective_all", None, "designs silent to six donors"),
    ("nbCrossSilent", "interferents/n_cross_silent", None, "designs silent to the other analytes"),
    ("nbFpDesigns", "interferents/n_fp_designs", None, "designs with a donor false positive"),
    ("nbFpImidazole", "interferents/fp_per_donor/imidazole", None, "false positives by imidazole"),
    ("nbFpAmine", "interferents/fp_per_donor/amine", None, "false positives by amine"),
    ("nbFpThiolate", "interferents/fp_per_donor/thiolate", None, "false positives by thiolate"),
    ("nbMarginMedian", "interferents/margin_median", None, "median selectivity margin"),
    ("nbMarginMin", "interferents/margin_min", None, "min selectivity margin"),
    ("nbMarginMax", "interferents/margin_max", None, "max selectivity margin"),
    ("nbMarginPos", "interferents/n_margin_pos", None, "designs with margin > 0"),
    ("nbMarginPointOne", "interferents/n_margin_gt_0p1", None, "designs with margin > 0.1"),
    ("nbMarginPointTwo", "interferents/n_margin_gt_0p2", None, "designs with margin > 0.2"),
    ("nbMarginNonPos", "interferents/n_margin_nonpos", None, "designs with margin <= 0"),
    ("nbFstarMin", "interferents/fstar_imidazole_min", None, "f* min (imidazole h)"),
    ("nbFstarMedian", "interferents/fstar_imidazole_median", None, "f* median"),
    ("nbFstarMax", "interferents/fstar_imidazole_max", None, "f* max"),
    ("nbRatioCl", "competition/cyanide/chloride", None, "K ratio vs chloride"),
    ("nbRatioHCO", "competition/cyanide/bicarbonate", None, "K ratio vs bicarbonate"),
    ("nbRatioPO", "competition/cyanide/phosphate", None, "K ratio vs phosphate"),
    ("nbRatioAlb", "competition/cyanide/albumin", None, "K ratio vs albumin"),
    ("nbRatioHis", "competition/cyanide/histidine", None, "K ratio vs histidine"),
    ("nbRatioCys", "competition/cyanide/cysteine", None, "K ratio vs cysteine"),
    ("nbTwoRT", "sizemap/two_RT_kJ", None, "2RT at 310 K (kJ/mol)"),
    ("nbDcritSix", "sizemap/table/6/d_c", None, "d_c, Gamma=6 kJ/mol (nm)"),
    ("nbDcritEight", "sizemap/table/8/d_c", None, "d_c, Gamma=8 (nm)"),
    ("nbDcritTen", "sizemap/table/10/d_c", None, "d_c, Gamma=10 (nm)"),
    ("nbLoopBulkEight", "sizemap/table/8/bulk", None, "bulk loop width, Gamma=8 (K)"),
    ("nbLoopFiftyEight", "sizemap/table/8/50", None, "50 nm loop width, Gamma=8 (K)"),
    ("nbLoopBulkSix", "sizemap/table/6/bulk", None, "bulk loop width, Gamma=6 (K)"),
    ("nbLoopBulkTen", "sizemap/table/10/bulk", None, "bulk loop width, Gamma=10 (K)"),
    ("nbMomentPerFe", "nv/moment_per_fe_muB", None, "Curie moment per HS Fe (mu_B)"),
    ("nbNFeFifty", "nv/n_fe_50nm", None, "Fe centres in 50 nm particle"),
    ("nbNvFiftyHundred", "nv/field_50nm_at_100nm_nT", None, "contrast 50 nm @ 100 nm (nT)"),
    ("nbNvHundredFifty", "nv/field_100nm_at_50nm_uT", None, "contrast 100 nm @ 50 nm (uT)"),
    ("nbNvTimeTyp", "nv/t_typ_50_100_s", None, "SNR-3 time, typical NV (s)"),
    ("nbNvTimeBest", "nv/t_best_50_100_s", None, "SNR-3 time, best NV (s)"),
    ("nbNvTimeHundredFifty", "nv/t_typ_100_50_s", None, "SNR-3 time 100 nm @ 50 nm (s)"),
    ("nbPitchFifty", "nv/pitch_nm/50", None, "array pitch, sensor at 50 nm"),
    ("nbPitchHundred", "nv/pitch_nm/100", None, "array pitch, sensor at 100 nm"),
    ("nbPitchTwoHundred", "nv/pitch_nm/200", None, "array pitch, sensor at 200 nm"),
    ("nbTauExchange", "response/tau_exchange_s", 1e6, "water exchange time (us)"),
    ("nbTauAssocTwenty", "response/tau_assoc_20uM_s", 1e3, "association at 20 uM (ms)"),
    ("nbTauAssocHundredNano", "response/tau_assoc_100nM_s", 1.0, "association at 100 nM (s)"),
    ("nbTauShellGel", "response/tau_shell_gel_s", 1e6, "hydrogel shell delay (us)"),
    ("nbTauShellSilica", "response/tau_shell_silica_s", 1e3, "silica shell delay (ms)"),
    ("nbIonPerGyFifty", "radiation/ion_per_gy_50nm", None, "ionisations/Gy, 50 nm"),
    ("nbGyPerIonFifty", "radiation/gy_per_ion_50nm", None, "Gy per ionisation, 50 nm"),
    ("nbIonPerGyTwoHundred", "radiation/ion_per_gy_200nm", None, "ionisations/Gy, 200 nm"),
    ("nbParticlesPerRadiograph", "radiation/particles_per_radiograph", None, "1 particle in N per 10 mGy"),
    ("nbFracPerFraction", "radiation/frac_per_2gy_fraction_pct", None, "% ionised per 2 Gy"),
    ("nbIonRadiotherapy", "radiation/ion_per_60gy_course", None, "ionisations per 60 Gy"),
    ("nbStoppingWater", "radiation/stopping_water_10MeV", None, "Bethe S/rho water 10 MeV"),
    ("nbStoppingSco", "radiation/stopping_sco_10MeV", None, "Bethe S/rho framework 10 MeV"),
    ("nbGyPerFluenceSco", "radiation/gy_per_1e10_fluence_sco", None, "Gy from 1e10 cm^-2"),
    ("nbDispFiftyTenMeV", "radiation/disp_50nm_10MeV_1e10", None, "displacements per 50 nm particle"),
    ("nbIonFiftyTenMeV", "radiation/ion_50nm_10MeV_1e10", None, "ionisations per 50 nm particle"),
    ("nbRatioIonDisp", "radiation/ratio_ion_disp", None, "ionisation/displacement ratio"),
    ("nbGrPhi", "graphene/full/dphi_mV", None, "Helmholtz shift (mV)"),
    ("nbGrDn", "graphene/full/dn_cm2", 1e-12, "induced density (1e12 cm^-2)"),
    ("nbGrRelLow", "graphene/full/rel_to_1e12_pct", None, "% of 1e12 cm^-2"),
    ("nbGrRelHigh", "graphene/full/rel_to_1e13_pct", None, "% of 1e13 cm^-2"),
    ("nbGrPhiHalf", "graphene/half/dphi_mV", None, "shift at 0.5 D (mV)"),
    ("nbGrDnHalf", "graphene/half/dn_cm2", 1e-12, "density at 0.5 D (1e12 cm^-2)"),
    ("mlRls", "geometry/R_LS", None, "Fe-N LS (A) [input]"),
    ("mlRhs", "geometry/R_HS", None, "Fe-N HS (A) [input]"),
    ("mlDR", "geometry/dR", None, "Fe-N lengthening (A)"),
    ("mlDvolPoly", "geometry/dvol_poly_pct", None, "octahedron volume change (%)"),
    ("mlAls", "geometry/a_LS", None, "a, LS (A)"),
    ("mlAhs", "geometry/a_HS", None, "a, HS (A)"),
    ("mlCls", "geometry/c_LS", None, "c, LS (A)"),
    ("mlChs", "geometry/c_HS", None, "c, HS (A)"),
    ("mlVls", "geometry/V_LS", None, "V per Fe, LS (A^3)"),
    ("mlVhs", "geometry/V_HS", None, "V per Fe, HS (A^3)"),
    ("mlDaPct", "geometry/da_pct", None, "a expansion (%)"),
    ("mlDcPct", "geometry/dc_pct", None, "c expansion (%)"),
    ("mlDvPct", "geometry/dV_pct", None, "volume expansion (%)"),
    ("mlFeLS", "geometry/Fe_per_nm3_LS", None, "Fe density LS (nm^-3)"),
    ("mlFeHS", "geometry/Fe_per_nm3_HS", None, "Fe density HS (nm^-3)"),
    ("mlVused", "geometry/V_used", None, "V per Fe at 2.6 nm^-3"),
    ("mlDtwoLS", "microscopy/d200_LS_nm", None, "d200 LS (nm)"),
    ("mlDtwoHS", "microscopy/d200_HS_nm", None, "d200 HS (nm)"),
    ("mlShiftTwo", "microscopy/shift_inplane_pct", None, "in-plane reflection shift (%)"),
    ("mlShiftZeroTwo", "microscopy/shift_pillar_pct", None, "pillar reflection shift (%)"),
    ("mlDq", "microscopy/dq_nm", None, "200 shift in q (nm^-1)"),
    ("mlDmin", "microscopy/d_min_nm", None, "Scherrer resolvability diameter (nm)"),
    ("mlRatioTwenty", "microscopy/ratio_20", None, "shift/width at 20 nm"),
    ("mlRatioFifty", "microscopy/ratio_50", None, "shift/width at 50 nm"),
    ("mlRatioFive", "microscopy/ratio_5", None, "shift/width at 5 nm"),
    ("mlPlanesTwenty", "microscopy/planes_20", None, "200 planes across 20 nm"),
    ("mlPlanesFifty", "microscopy/planes_50", None, "200 planes across 50 nm"),
    ("mlSurfTwenty", "microscopy/surf_pct_20", None, "% centres in outer 1 nm, 20 nm"),
    ("mlSurfFifty", "microscopy/surf_pct_50", None, "% centres in outer 1 nm, 50 nm"),
    ("mgDone", "microscopy/fringe_diff_one_nm", None, "diameter for 1-fringe difference"),
    ("mgDtwo", "microscopy/fringe_diff_two_nm", None, "diameter for 2-fringe difference"),
    ("mgFringesFifty", "microscopy/fringe_diff_50", None, "fringe difference at 50 nm"),
    ("mgFringesTwenty", "microscopy/fringe_diff_20", None, "fringe difference at 20 nm"),
]

NOT_REIMPLEMENTED = {
    "agreePTc": "device-calibrated surrogate agreement (companion paper; needs noisy-device simulation)",
    "rmsPTc": "held-out error of the device-calibrated surrogate (companion paper)",
    "nCal": "number of device calibration points (companion paper, an input there)",
}
INPUTS = {"mlInfoLimit": "information limit (input)", "mgMfp": "mean free path (input)",
          "mlFeUsed": "assumed Fe density (input)"}


def parse_table(tex, name):
    m = re.search(r"%% begin tab_" + name + r"\n(.*?)%% end tab_" + name, tex, re.S)
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip().rstrip("\\").strip()
        if not line or line.startswith("\\"):
            continue
        rows.append([c.strip() for c in line.split("&")])
    return rows


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer() and abs(v) > 50):
        return f"{v:,.0f}"
    if abs(v) < 0.01:
        return f"{v:.3g}"
    return f"{v:.4g}"


def main(tex_path=DEFAULT_TEX):
    tex = Path(tex_path).read_text()
    macros = parse_macros(tex)
    R = json.loads((ROOT / "results" / "results.json").read_text())
    lines = []
    counts = {"match": 0, "near": 0, "MISMATCH": 0}
    lines.append("| macro | quantity | paper | this reimplementation | verdict |")
    lines.append("|---|---|---|---|---|")
    for macro, path, scale, desc in MAP:
        ptxt = macros.get(macro)
        pval, ptxt_num = to_number(ptxt) if ptxt is not None else (None, None)
        ours = get(R, path)
        if scale is not None and ours is not None:
            ours = ours * scale
        v = verdict(ours, pval, ptxt_num) if pval is not None else "n/a"
        counts[v] = counts.get(v, 0) + 1
        lines.append(f"| `{macro}` | {desc} | {ptxt} | {fmt(ours)} | {v} |")
    # string-valued macro
    fps = R["interferents"]["fp_scaffolds"]
    paper_fps = macros.get("nbFpScaffolds", "")
    ok = all(s in paper_fps.replace("$+$", "+") for s in fps) and len(fps) == 2
    counts["match" if ok else "MISMATCH"] += 1
    lines.append(f"| `nbFpScaffolds` | false-positive scaffolds | {paper_fps} | {' and '.join(fps)} | {'match' if ok else 'MISMATCH'} |")

    # tables
    tab_lines = []
    tcounts = {"match": 0, "near": 0, "MISMATCH": 0}

    def add(tname, key, ptxt, ours):
        pval, pt = to_number(ptxt)
        v = verdict(ours, pval, pt)
        tcounts[v] = tcounts.get(v, 0) + 1
        tab_lines.append(f"| {tname} | {key} | {ptxt} | {fmt(ours)} | {v} |")

    for row in parse_table(tex, "sizemap"):
        lab = row[0]
        for g, cell in zip(("6", "8", "10"), row[1:]):
            if lab.startswith("critical"):
                add("sizemap", f"d_c, Gamma={g}", cell, R["sizemap"]["table"][g]["d_c"])
            else:
                add("sizemap", f"d={lab}, Gamma={g}", cell, R["sizemap"]["table"][g][lab])
    for row in parse_table(tex, "nv"):
        d = row[0]
        for s, cell in zip(("10", "50", "100", "200"), row[1:]):
            add("nv", f"d={d} nm, standoff {s} nm", cell, R["nv"]["table_nT"][d][s])
    comp_keys = ["cyanide", "azide", "thiocyanate", "chloride"]
    for row in parse_table(tex, "competition"):
        donor = re.sub(r"\s*\(.*", "", row[0]).replace(", intracellular", " (intracellular)")
        for X, cell in zip(comp_keys, row[1:]):
            cell = cell.replace("$", "")
            add("competition", f"{donor} / {X}", cell, R["competition"][X][donor])

    head = [
        "# VALIDATION: paper-reported quantities vs this reimplementation",
        "",
        "Generated by `scripts/validate.py` from `results/results.json`. Paper values are read from",
        "the manuscript's `\\newcommand` block and table bodies for comparison only.",
        "Verdict rule: **match** = our value rounded to the paper's printed precision equals the",
        "paper value; **near** = within one unit of the last printed digit or 1 %; **MISMATCH** otherwise.",
        "Mismatches are reported as found; no unstated parameter was tuned to remove them",
        "(see ASSUMPTIONS.md for every choice the paper leaves open, and the discussion below).",
        "",
        f"Scalar macros: {counts['match']} match, {counts['near']} near, {counts['MISMATCH']} mismatch.",
        f"Table cells: {tcounts['match']} match, {tcounts['near']} near, {tcounts['MISMATCH']} mismatch.",
        "",
        "## Scalar quantities (macros used in the paper body)",
        "",
    ]
    tail = ["", "## Table cells", "", "| table | entry | paper | this reimplementation | verdict |",
            "|---|---|---|---|---|"] + tab_lines
    tail += ["", "## Not reimplemented / inputs", ""]
    for k, v in NOT_REIMPLEMENTED.items():
        tail.append(f"- `{k}` = {macros.get(k)}: {v}. Not computed here.")
    for k, v in INPUTS.items():
        tail.append(f"- `{k}` = {macros.get(k)}: {v}; used as given.")
    notes = (ROOT / "scripts" / "validation_notes.md")
    if notes.exists():
        tail += ["", notes.read_text()]
    (ROOT / "VALIDATION.md").write_text("\n".join(head + lines + tail) + "\n")
    print(f"scalars: {counts}; tables: {tcounts}")
    for l in lines + tab_lines:
        if "MISMATCH" in l or "| near |" in l:
            print(l)


if __name__ == "__main__":
    main(*sys.argv[1:])
