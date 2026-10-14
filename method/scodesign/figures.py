"""Figures analogous to the manuscript's figures 1, 3, 4, 5, 6(a) and 7, drawn from results.json.
Saved as figures/reimpl_fig*.pdf|png (never the manuscript's own file names)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import constants as C

PLAT = list(C.PLATFORMS.keys())
# fixed categorical order (one slot per platform) + marker shape as secondary encoding
COL = dict(zip(PLAT, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MRK = dict(zip(PLAT, ["o", "s", "D", "^", "v"]))
INK = "#333333"
MUTED = "#8a8a8a"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False})


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1(R, out):
    s = R["sweep"]; M = R["macros"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.array(s["x"])
    ax[0].plot(x, s["exact"], color=INK, label="exact")
    ax[0].plot(x, s["mf"], color=INK, ls="--", lw=1.5, label="mean field")
    ax[0].plot(x, s["pt"], color=INK, ls=":", lw=1.5, label="mean field + PT2")
    ax[0].axhline(0, color=MUTED, lw=0.8)
    for key in ("xoExact", "xoMF", "xoPT"):
        ax[0].plot([M[key]], [0], "o", mfc="white", mec=INK, ms=7)
    ax[0].set_xlabel(r"$\Delta/B$ (B = 700 cm$^{-1}$)"); ax[0].set_ylabel(r"$E_{HS}-E_{LS}$ (cm$^{-1}$)")
    ax[0].legend(); ax[0].set_title("(a) spin-state gap", loc="left")
    T = np.array(R["T_exact"])
    n_six = len(C.SIXTH_SITE)
    six = np.arange(len(T)) % n_six
    ok_cplx = np.zeros(len(T), bool)
    rng = np.random.default_rng(0)
    # successful pairs: resting T<=290 and analyte T>=330 (or reverse)
    for sc in range(len(T) // n_six):
        tr = T[sc * n_six]
        for xi in range(1, n_six):
            ta = T[sc * n_six + xi]
            if min(tr, ta) <= C.T_LO and max(tr, ta) >= C.T_HI:
                ok_cplx[sc * n_six + xi] = True
    for xi, name in enumerate(C.SIXTH_SITE):
        sel = six == xi
        y = xi + rng.uniform(-0.3, 0.3, sel.sum())
        okk = ok_cplx[sel]
        ax[1].scatter(T[sel][~okk], y[~okk], s=6, facecolors="none", edgecolors=MUTED, lw=0.5)
        ax[1].scatter(T[sel][okk], y[okk], s=8, color="#2a78d6")
    ax[1].axvspan(C.T_LO, C.T_HI, color="#eeeeee", zorder=0)
    ax[1].set_yticks(range(n_six)); ax[1].set_yticklabels(["H2O (rest)"] + C.SIXTH_SITE[1:])
    ax[1].set_xlabel(r"$T_{1/2}$ exact (K)"); ax[1].set_xlim(-1500, 800)
    ax[1].set_title(f"(b) design landscape: {M['nOkExact']} successful pairs (filled)", loc="left")
    _save(fig, out, "reimpl_fig1")


def fig3(R, out):
    tab = R["tab_noise"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    si = 1
    for k, p in enumerate(PLAT):
        rows = [r for r in tab if r["platform"] == p]
        r = rows[si]
        ax[0].bar(k - 0.2, r["mit_rms"], 0.38, color=COL[p])
        ax[0].bar(k + 0.2, r["raw_rms"], 0.38, facecolor="white", edgecolor=COL[p], lw=1.5)
        ax[0].plot([k - 0.35, k - 0.05], [abs(r["mit_bias"])] * 2, color=INK, lw=1)
        ax[0].plot([k + 0.05, k + 0.35], [abs(r["raw_bias"])] * 2, color=INK, lw=1)
        sh = [x["shots"] for x in rows]
        ax[1].plot(sh, [x["mit_rms"] for x in rows], marker=MRK[p], color=COL[p], label=C.PLATFORMS[p]["label"])
    raws = [r["raw_rms"] for r in tab]
    ax[1].axhspan(min(raws), max(raws), color="#eeeeee", zorder=0)
    ax[0].axhline(20, color=INK, ls="--", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_xticks(range(len(PLAT))); ax[0].set_xticklabels(PLAT)
    ax[0].set_ylabel("RMS error in T (K), 4,000 shots"); ax[0].set_title("(a) filled: mitigated, open: raw; tick = |bias|", loc="left")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("shots per basis"); ax[1].set_ylabel("mitigated RMS (K)")
    ax[1].legend(fontsize=7); ax[1].set_title("(b) mitigated error vs shots (band: raw arms)", loc="left")
    _save(fig, out, "reimpl_fig3")


def fig4(R, out):
    M = R["macros"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cp = R["cal_points"]
    ax[0].scatter(cp["E2"], cp["Ecorr_dev"], marker="s", s=12, color="#2a78d6", label="calibration points (device)")
    x = np.linspace(min(cp["E2"]), max(cp["E2"]), 10)
    ax[0].plot(x, M["corrA"] + M["corrB"] * x, color=INK, label="least-squares recalibration")
    ax[0].set_xlabel(r"$E^{(2)}$ (cm$^{-1}$)"); ax[0].set_ylabel(r"$E_{corr}$ (cm$^{-1}$)")
    ax[0].legend(fontsize=7); ax[0].set_title("(a) device as calibrator", loc="left")
    for p in PLAT:
        rows = [r for r in R["tab_surr"] if r["platform"] == p and r["arm"] == "mit"]
        ax[1].plot([r["shots"] for r in rows], [r["rms"] for r in rows], marker=MRK[p], color=COL[p],
                   label=C.PLATFORMS[p]["label"])
    ca = R["cal_arm"]
    ax[1].plot([ca["shots"]], [M["rmsPTc"]], "o", ms=14, mfc="none", mec=INK)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("shots per basis")
    ax[1].set_ylabel("held-out RMS of surrogate (K)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) calibration arm (mitigated)", loc="left")
    _save(fig, out, "reimpl_fig4")


def fig5(R, out):
    M = R["macros"]
    T2 = np.array(R["T2"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.arange(np.floor(T2.min() / 20) * 20, 800, 20)
    ax[0].hist(np.clip(T2, bins[0], 799), bins=bins, color="#2a78d6", rwidth=0.9)
    ax[0].set_yscale("symlog")
    for t in (C.T_LO, C.T_HI):
        ax[0].axvline(t, color=INK, ls="--", lw=0.8)
        ax[0].axvspan(t - M["band"], t + M["band"], color="#eeeeee", zorder=0)
    ax[0].set_xlabel("depth-2 $T_{1/2}$ (K)"); ax[0].set_ylabel("complexes")
    ax[0].set_title(f"(a) dispatch band ±{M['band']:.0f} K: {M['nDispatched']} dispatched", loc="left")
    pol = R["tab_policies"]
    for p in PLAT:
        if C.PLATFORMS[p]["currency"] is None:
            continue
        for arm, ls, fill in (("mit", "-", True), ("raw", ":", False)):
            rows = [r for r in pol if r.get("platform") == p and r.get("arm") == arm]
            ax[1].plot([r["cost"] for r in rows], [r["rms"] for r in rows], ls=ls, marker=MRK[p], color=COL[p],
                       mfc=COL[p] if fill else "white", lw=1.2,
                       label=f"{C.PLATFORMS[p]['label']} ({C.PLATFORMS[p]['currency']})" if arm == "mit" else None)
    rt = pol[-1]
    cst = rt["cost_hqc"] + rt["cost_usd"]
    ax[1].plot([cst], [rt["rms"]], "*", ms=14, color=INK, label="routed")
    ax[1].axhline(M["clRmsePTc"], color=INK, ls="--", lw=0.8, label="calibrated surrogate")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("cost of dispatched set (USD or HQC)")
    ax[1].set_ylabel("RMS over dispatched (K)"); ax[1].legend(fontsize=6)
    ax[1].set_title("(b) uniform policies vs routed", loc="left")
    _save(fig, out, "reimpl_fig5")


def fig6(R, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for p in ("Helios", "Heron", "Cepheus"):
        for arm, ls in (("mit", "-"), ("rsc", "--")):
            r = [b for b in R["budget_law"] if b["platform"] == p and b["arm"] == arm and b["shots"] == 1000][0]
            n = np.array(r["n"])
            ax.plot(n, np.sqrt(r["mse"]), ls=ls, marker=MRK[p], color=COL[p],
                    label=f"{p} {arm}: b={r['floor']:.0f} K, s={r['per_point']:.0f} K")
            nn = np.linspace(n.min(), n.max(), 100)
            ax.plot(nn, np.sqrt(r["floor"] ** 2 + r["per_point"] ** 2 / nn), color=COL[p], lw=0.6, ls=ls)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("calibration complexes")
    ax.set_ylabel("held-out surrogate RMS (K)"); ax.legend(fontsize=6)
    ax.set_title("budget law at 1,000 shots (lines: fitted b² + s²/n)", loc="left")
    _save(fig, out, "reimpl_fig6a")


def fig7(R, out):
    rg = R["roughness"]
    sig = np.array(rg["sigmas"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for p in PLAT:
        c = [x for x in rg["combos"] if x["platform"] == p and x["arm"] == "rsc" and x["shots"] == 16000]
        if not c:
            continue
        c = c[0]
        ax[0].plot(sig, c["curve"], marker=MRK[p], color=COL[p], ms=4, label=C.PLATFORMS[p]["label"])
        ax[0].axhline(c["D"], color=COL[p], ls=":", lw=1)
        if np.isfinite(c["sigma_star"]):
            ax[0].plot([c["sigma_star"]], [c["D"]], "*", color=COL[p], ms=11, mec=INK, mew=0.5)
    ax[0].set_xscale("symlog", linthresh=10); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\sigma_{rough}$ (K)"); ax[0].set_ylabel("surrogate held-out RMS (K)")
    ax[0].legend(fontsize=6); ax[0].set_title("(a) reference-scaled, 16k shots", loc="left")
    ok = [c for c in rg["combos"] if np.isfinite(c["sigma_star"]) and c["sigma_star"] > 0]
    for p in PLAT:
        cc = [c for c in ok if c["platform"] == p]
        ax[1].scatter([c["predicted"] for c in cc], [c["sigma_star"] for c in cc], marker=MRK[p], color=COL[p], s=18,
                      label=p)
    lim = [5, 800]
    ax[1].plot(lim, lim, color=INK, lw=0.8); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"predicted $\sqrt{D^2-b^2-s^2/n}$ (K)"); ax[1].set_ylabel(r"sweep crossing $\sigma^*$ (K)")
    ax[1].legend(fontsize=6); ax[1].set_title("(b) closed form vs sweep", loc="left")
    if "decisions_16k" in rg:
        d = rg["decisions_16k"]
        ws = np.array([x["wrong_surr"] for x in d]); wd = np.array([x["wrong_dev"] for x in d])
        for arr, lab, col in ((ws, "calibrate", "#2a78d6"), (wd, "evaluate", "#eb6834")):
            m, s = arr.mean(0), arr.std(0)
            ax[2].plot(sig, m, color=col, label=lab); ax[2].fill_between(sig, m - s, m + s, color=col, alpha=0.2, lw=0)
        ax[2].set_xscale("symlog", linthresh=10); ax[2].set_xlabel(r"$\sigma_{rough}$ (K)")
        ax[2].set_ylabel(f"wrong decisions / {R['macros']['nPairs']}"); ax[2].legend(fontsize=7)
        ax[2].set_title("(c) decisions at 16k shots (mean ± sd over arms)", loc="left")
    _save(fig, out, "reimpl_fig7")


def make_all(results_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(results_path) as f:
        R = json.load(f)
    for fn in (fig1, fig3, fig4, fig5, fig6, fig7):
        fn(R, out_dir)
    print(f"[figures] written to {out_dir}")


# ------------------------------------------------------------------ implemented stack (sec. 6.3-6.7)
def fig6_stack(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = R["tab_rdm"]
    for p in PLAT:
        rr = [r for r in rows if r["platform"] == p]
        ax[0].plot([r["shots"] for r in rr], [r["rdm"] for r in rr], marker=MRK[p], color=COL[p], label=C.PLATFORMS[p]["label"])
        ax[0].plot([r["shots"] for r in rr], [r["rsc"] for r in rr], ls=":", color=COL[p], lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("shots per basis")
    ax[0].set_ylabel("RMS over dispatched (K)"); ax[0].legend(fontsize=6)
    ax[0].set_title("(b) decoder (solid) vs ref. scaled (dotted)", loc="left")
    g = R["generative"]["curves"]
    n = np.array(g["n"])
    for key, col, lab in (("uniform", MUTED, "uniform"), ("prior", "#eb6834", "site-composition prior"),
                          ("scorer", "#2a78d6", "logistic scorer")):
        c = np.array(g[key], float)
        ax[1].plot(n[:len(c)], c / n[:len(c)], color=col, label=lab)
    ax[1].set_xlabel("distinct proposals"); ax[1].set_ylabel("hit rate (exact solver)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(c) four- and five-ligand space", loc="left")
    for p, mk in (("Helios", "^"), ("Heron", "s")):
        e = R["escalation"][p]
        cfix = [x["cost"] for x in e["fixed"]]; c0 = cfix[-1]
        ax[2].plot([x["cost"] / c0 for x in e["fixed"]], [x["err"] for x in e["fixed"]], ls="--", marker=mk,
                   color=COL[p], mfc="white", label=f"{p} fixed")
        ax[2].plot([x["cost"] / c0 for x in e["learned"]], [x["err"] for x in e["learned"]], marker=mk,
                   color=COL[p], label=f"{p} Q-learned")
    ax[2].set_xscale("log"); ax[2].set_xlabel("cost / cost of full escalation"); ax[2].set_ylabel("misclassification rate")
    ax[2].legend(fontsize=7); ax[2].set_title("(d) escalation policy (held-out epochs)", loc="left")
    _save(fig, out, "reimpl_fig6bcd")


def fig8_qaoa(R, out):
    q = R["qaoa"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    E = np.array(q["landscape"]["E"])
    im = ax[0].imshow(E, origin="lower", aspect="auto", cmap="Blues",
                      extent=[0, np.pi, 0, np.pi / 2])
    fig.colorbar(im, ax=ax[0], label="<f>")
    ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$\beta$"); ax[0].set_title("(a) p = 1 landscape", loc="left")
    ps = sorted(q["depths"], key=int)
    ax[1].plot([int(p) for p in ps], [q["depths"][p]["ratio"] for p in ps], marker="o", color="#2a78d6",
               label="approximation ratio <f>/f_max")
    ax[1].plot([int(p) for p in ps], [q["depths"][p]["p_opt"] for p in ps], marker="s", color="#eb6834",
               label="P(optimal set)")
    ax[1].set_xlabel("QAOA depth p"); ax[1].set_xticks([1, 2, 3]); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) twelve-qubit dispatch QAOA (ideal)", loc="left")
    _save(fig, out, "reimpl_fig8_qaoa")


_make_all_base = make_all


def make_all(results_path, out_dir):
    _make_all_base(results_path, out_dir)
    with open(results_path) as f:
        R = json.load(f)
    if "tab_rdm" in R:
        fig6_stack(R, out_dir)
    if "qaoa" in R:
        fig8_qaoa(R, out_dir)
