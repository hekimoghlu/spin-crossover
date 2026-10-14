#!/usr/bin/env python3
"""Regenerate figures analogous to the manuscript's figures 1-4 (plus a Monte
Carlo figure from the Rust component).  Output: figures/reimpl_fig*.pdf/png.
All micrographs, potentials and diffraction patterns are SIMULATED."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scmaterials import (geometry, ligandfield as lf, micrograph, radiation, readout,  # noqa: E402
                         scattering, sizemap)

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((ROOT / "results" / "results.json").read_text())
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5})
SIM = dict(fontsize=6, color="0.25", ha="right", va="top")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def tag(ax, s):
    ax.set_title(f"({s})", loc="left", fontweight="bold")


# ---------------------------------------------------------------- figure 1
def fig1():
    fig = plt.figure(figsize=(7.2, 2.6))
    ax = fig.add_subplot(1, 3, 1)
    tag(ax, "a")
    for st, col, off in (("LS", "C0", -3.2), ("HS", "C3", 3.2)):
        r = geometry.R_FEN[st]
        pts = geometry.coordination_sphere(st)
        # oblique projection
        P = pts[:, :2] + 0.45 * pts[:, 2:3] * np.array([0.6, 0.45])
        for p in P:
            ax.plot([off, off + p[0]], [0, p[1]], color=col, lw=1.5)
        ax.scatter(off + P[:5, 0], P[:5, 1], s=40, color="tab:blue", zorder=3)
        ax.scatter(off + P[5, 0], P[5, 1], s=40, color="tab:orange", zorder=3)
        ax.scatter([off], [0], s=80, color="tab:brown", zorder=4)
        ax.text(off, -3.0, f"{st}: Fe-N {r:.2f} A", ha="center")
    ax.text(0, 2.9, f"polyhedron +{R['geometry']['dvol_poly_pct']:.0f} %", ha="center")
    ax.set_xlim(-6, 6); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    tag(ax, "b")
    B = 700.0
    r = np.linspace(14, 24, 200)
    ex = [lf.dH_exact(x * B, B) for x in r]
    mf = [lf.dH_meanfield(x * B, B) for x in r]
    ax.plot(r, np.array(ex) / 1e3, "k-", label="exact (5-orbital)")
    ax.plot(r, np.array(mf) / 1e3, "k--", label="mean field")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(R["design_space"]["crossover_exact_DoverB"], color="C0", lw=0.6, ls=":")
    ax.set_xlabel(r"$\Delta/B$ (B = 700 cm$^{-1}$)")
    ax.set_ylabel(r"$E_{HS}-E_{LS}$ ($10^3$ cm$^{-1}$)")
    ax.text(14.3, 6, r"LS $t_{2g}^6e_g^0$", fontsize=7)
    ax.text(14.3, -8, r"HS $t_{2g}^4e_g^2$", fontsize=7)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(1, 3, 3)
    tag(ax, "c")
    atoms, cell = geometry.hofmann_atoms("LS")
    a, c = cell["a"], cell["c"]
    cols = {"Fe": "tab:brown", "Pt": "0.5", "C": "k", "N": "tab:blue", "H": "0.85"}
    sizes = {"Fe": 60, "Pt": 70, "C": 15, "N": 18, "H": 6}
    for el, p in atoms:
        q = (p[0] + 0.35 * p[1], p[2] + 0.25 * p[1])
        ax.scatter(*q, s=sizes[el], color=cols[el], zorder=3 if el in ("Fe", "Pt") else 2)
    box = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0], [0, 0, 0], [0, 0, c], [a, 0, c], [a, a, c], [0, a, c], [0, 0, c]])
    ax.plot(box[:, 0] + 0.35 * box[:, 1], box[:, 2] + 0.25 * box[:, 1], color="0.6", lw=0.6)
    ax.text(0, -1.6, f"a = {a:.2f} A, c = {c:.2f} A (LS)\nV/Fe = {cell['V']:.0f} A$^3$; +{R['geometry']['dV_pct']:.1f} % on LS->HS", fontsize=6.5)
    ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Reimplementation: testbed geometry from tabulated bond lengths", fontsize=8, y=1.06)
    save(fig, "reimpl_fig1")


# ---------------------------------------------------------------- figure 2
def fig2():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axs[0, 0]; tag(ax, "a")
    m = np.array(R["interferents"]["margins_sorted"])[::-1]
    ax.bar(np.arange(len(m)), m, width=1.0, color=np.where(m > 0, "#5b4ea6", "#d9534f"))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("successful design, ranked by selectivity margin")
    ax.set_ylabel(r"margin $f^\ast - f_{donor}$ (tightest donor)")
    ax.text(5, -0.09, f"{R['interferents']['n_margin_nonpos']} designs <= 0; "
            f"{R['interferents']['n_fp_designs']} read as analyte", fontsize=6.5, color="#d9534f")

    ax = axs[0, 1]; tag(ax, "b")
    d = np.logspace(np.log10(5), np.log10(1000), 300)
    for g, ls in ((6e3, "-."), (8e3, "--"), (10e3, "-")):
        w = [sizemap.loop_width(float(sizemap.gamma_of_d(x, g))) for x in d]
        dc = sizemap.d_critical(g)
        ax.plot(d, w, ls, label=rf"$\Gamma_{{bulk}}$ = {g/1e3:g} kJ mol$^{{-1}}$")
        ax.plot([dc], [0], "o", ms=3, color="k")
        ax.text(dc, 2 + g / 1e3, f"{dc:.0f} nm", fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("particle diameter d (nm)"); ax.set_ylabel("loop width (K)")
    ax.legend(loc="upper left")

    ax = axs[1, 0]; tag(ax, "c")
    dd = np.array([20, 50, 100, 200])
    for s, mk in ((10, "o"), (50, "s"), (100, "^"), (200, "v")):
        ax.plot(dd, readout.stray_field_contrast(dd, s) * 1e9, marker=mk, label=f"{s} nm standoff")
    ax.axhline(3e3, color="0.4", ls="--", lw=0.7); ax.text(21, 3.6e3, "typical NV, 1 s (SNR 3)", fontsize=6)
    ax.axhline(30, color="0.4", ls=":", lw=0.7); ax.text(21, 36, "best NV, 1 s (SNR 3)", fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("particle diameter (nm)")
    ax.set_ylabel("HS-LS stray-field contrast (nT)"); ax.legend(loc="lower right")

    ax = axs[1, 1]; tag(ax, "d")
    dd = np.logspace(np.log10(20), np.log10(200), 50)
    ax.plot(dd, radiation.ionisations_per_gray(dd), color="#5b4ea6")
    for v, lab in ((1 / 0.010, "one per particle per radiograph (10 mGy)"), (1 / 2.0, "per fraction (2 Gy)"),
                   (1 / 60.0, "per course (60 Gy)")):
        ax.axhline(v, color="0.4", ls=":", lw=0.7); ax.text(21, v * 1.2, lab, fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("particle diameter (nm)")
    ax.set_ylabel("ionisations per particle per gray")
    fig.suptitle("Reimplementation: computed design maps", fontsize=8)
    fig.tight_layout()
    save(fig, "reimpl_fig2")


# ---------------------------------------------------------------- figure 3
def fig3():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax, zone, t in ((axs[0, 0], "001", "a"), (axs[0, 1], "100", "b")):
        tag(ax, t)
        x, V = scattering.projected_potential("LS", zone, extent_nm=1.0, npix=200)
        ax.imshow(V, extent=[x[0], x[-1], x[0], x[-1]], origin="lower", cmap="gray")
        ax.set_xlabel("nm"); ax.set_ylabel("nm")
        ax.set_title(f"SIMULATED projected potential [{zone}], 0.08 nm", fontsize=6.5, loc="right")
    ax = axs[1, 0]; tag(ax, "c")
    q = np.linspace(1.2, 3.6, 1200)
    for st, col in (("LS", "C0"), ("HS", "C3")):
        p20 = scattering.radial_profile(st, 20.0, q)
        p5 = scattering.radial_profile(st, 5.0, q)
        norm = max(scattering.radial_profile("LS", 20.0, q).max(), 1)
        ax.plot(q, p20 / norm, color=col, lw=0.9, label=f"{st}, 20 nm")
        ax.fill_between(q, 0, p5 / norm, color=col, alpha=0.2, label=f"{st}, 5 nm")
    ax.set_xlabel(r"scattering vector 1/d (nm$^{-1}$)"); ax.set_ylabel("simulated intensity (norm.)")
    ax.legend(); ax.text(3.55, 0.98, "simulated (kinematic)", **SIM)
    ax = axs[1, 1]; tag(ax, "d")
    d = np.logspace(0, 2, 100)
    ax.loglog(d, 0.9 / d, "k-", label="Scherrer peak width 0.9/d")
    dq = R["microscopy"]["dq_nm"]
    ax.axhline(dq, color="C3", ls="--", label=f"200 shift {dq:.3f} nm$^{{-1}}$")
    ax.axvline(R["microscopy"]["d_min_nm"], color="k", ls=":", lw=0.8)
    ax.text(R["microscopy"]["d_min_nm"] * 1.05, 0.5, f"{R['microscopy']['d_min_nm']:.1f} nm", fontsize=7)
    ax.set_xlabel("particle diameter (nm)"); ax.set_ylabel(r"width or shift (nm$^{-1}$)"); ax.legend()
    fig.suptitle("Reimplementation: SIMULATED microscopy of the model framework", fontsize=8)
    fig.tight_layout()
    save(fig, "reimpl_fig3")


# ---------------------------------------------------------------- figure 4
def fig4():
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])
    ens = []
    for k, (dm, field) in enumerate(((5, 70), (20, 260), (50, 640))):
        ax = fig.add_subplot(gs[0, k])
        I, ds, px = micrograph.bright_field(dm, field, npix=256, n_target=45, seed=10 + k)
        ens.append(ds)
        ax.imshow(I, cmap="gray", vmin=0, vmax=1.3, extent=[0, field, 0, field], origin="lower")
        ax.set_title(f"({'abc'[k]}) d = {dm} nm", loc="left", fontweight="bold")
        ax.text(field * 0.97, field * 0.97, "SIMULATED", fontsize=6, color="w", ha="right", va="top",
                bbox=dict(fc="k", alpha=0.5, lw=0))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"field {field} nm")
    ax = fig.add_subplot(gs[1, 0])
    img, fft, px = micrograph.lattice_particle(d_nm=12.0, field_nm=16.0, npix=320)
    ax.imshow(img, cmap="gray", extent=[0, 16, 0, 16], origin="lower")
    ax.set_title("(d) [001], 12 nm", loc="left", fontweight="bold")
    ax.text(15.5, 15.5, "SIMULATED", fontsize=6, color="w", ha="right", va="top", bbox=dict(fc="k", alpha=0.5, lw=0))
    ax.set_xticks([]); ax.set_yticks([])
    ins = ax.inset_axes([0.0, 0.0, 0.38, 0.38])
    c = fft.shape[0] // 2; w = 60
    ins.imshow(fft[c - w:c + w, c - w:c + w], cmap="gray"); ins.set_xticks([]); ins.set_yticks([])
    ax = fig.add_subplot(gs[1, 1:])
    ax.set_title("(e) diameter distributions", loc="left", fontweight="bold")
    for ds, col, lab in zip(ens, ("C0", "C3", "C2"), ("5 nm", "20 nm", "50 nm")):
        ax.hist(ds, bins=np.linspace(0, 90, 46), histtype="step", color=col, label=f"mean {lab}")
    for g in ("8", "6"):
        ax.axvline(R["sizemap"]["table"][g]["d_c"], color="k", ls="--", lw=0.7)
        ax.text(R["sizemap"]["table"][g]["d_c"] + 0.5, ax.get_ylim()[1] * 0.9, f"d_c ({g} kJ/mol)", fontsize=6)
    ax.axvline(R["microscopy"]["d_min_nm"], color="tab:orange", ls=":", lw=0.9)
    ax.set_xlabel("particle diameter (nm)"); ax.set_ylabel("count"); ax.legend()
    ax = fig.add_subplot(gs[2, :])
    ax.set_title("(f) counting fringes: LS - HS 200-fringe difference", loc="left", fontweight="bold")
    d = np.linspace(0, 60, 200)
    ax.plot(d, d * R["microscopy"]["dq_nm"], color="C3")
    ax.axhline(1, color="0.5", ls=":"); ax.axhline(2, color="0.5", ls=":")
    ax.axvline(R["microscopy"]["fringe_diff_one_nm"], color="k", ls=":", lw=0.7)
    ax.axvline(R["microscopy"]["d_min_nm"], color="tab:orange", ls="--", lw=0.7)
    ax.set_xlabel("particle diameter (nm)"); ax.set_ylabel("fringe difference")
    fig.suptitle("Reimplementation: SIMULATED electron micrographs of particle ensembles", fontsize=8)
    fig.tight_layout()
    save(fig, "reimpl_fig4")


# ---------------------------------------------------------------- figure 5 (Rust)
def fig5():
    mc_csv = ROOT / "results" / "rust_mc.csv"
    loops = ROOT / "results" / "rust_mc.loops.csv"
    mf_csv = ROOT / "results" / "rust_mf.csv"
    if not mc_csv.exists():
        print("skip fig5: run the Rust component first")
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]; tag(ax, "a")
    rows = list(csv.DictReader(open(loops)))
    for d, col in (("6", "C0"), ("10", "C1"), ("30", "C3")):
        sel = [r for r in rows if r["gamma_bulk_kJmol"] == "12" and r["d_nm"] == d]
        T = [float(r["T"]) for r in sel]
        ax.plot(T, [float(r["x_heat"]) for r in sel], color=col, label=f"{d} nm")
        ax.plot(T, [float(r["x_cool"]) for r in sel], color=col, ls="--")
    ax.set_xlabel("T (K)"); ax.set_ylabel("HS fraction"); ax.legend(title=r"MC, $\Gamma$=12 kJ/mol")
    ax = axs[1]; tag(ax, "b")
    rows = list(csv.DictReader(open(mc_csv)))
    mf = list(csv.DictReader(open(mf_csv)))
    for g, col in (("8", "C0"), ("10", "C1"), ("12", "C3")):
        sel = [r for r in rows if r["gamma_bulk_kJmol"] == g]
        ax.plot([float(r["d_nm"]) for r in sel], [float(r["width_K"]) for r in sel], "o-", ms=3, color=col,
                label=f"MC {g} kJ/mol")
        msel = [r for r in mf if r["gamma_bulk_kJmol"] == g and r["d_nm"] != "inf"]
        if msel:
            msel = sorted(msel, key=lambda r: float(r["d_nm"]))
            ax.plot([float(r["d_nm"]) for r in msel], [float(r["loop_width_K"]) for r in msel], "-", lw=0.7,
                    color=col, alpha=0.6, label=f"mean field {g}")
    ax.set_xlim(0, 32); ax.set_ylim(-5, 60); ax.set_xlabel("particle diameter (nm)"); ax.set_ylabel("loop width (K)"); ax.legend(ncol=2)
    fig.suptitle("Reimplementation (extra): Ising-like Monte Carlo loops from the Rust component", fontsize=8)
    fig.tight_layout()
    save(fig, "reimpl_fig5")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
