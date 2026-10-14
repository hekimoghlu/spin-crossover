"""Coordination geometry and Hofmann-type Fe(pz)[Pt(CN)4] cell from bond lengths.

Rigid-linkage construction (paper section 'testbed complex'):
  Fe...Pt = Fe-N + C#N + Pt-C;  a = sqrt(2) * Fe...Pt;  c = 2 Fe-N(pz) + pz span.
Bond lengths other than Fe-N are standard values chosen here (ASSUMPTIONS.md).
Lengths in Angstrom.
"""
from __future__ import annotations

import numpy as np

R_FEN = {"LS": 1.97, "HS": 2.17}   # paper: tabulated Fe(II)-N
R_CN = 1.15                         # assumed: C#N in cyanometallates
R_PTC = 1.99                        # assumed: Pt-C in tetracyanoplatinate
# pyrazine ring (gas-phase electron diffraction values; assumed)
PZ_CN = 1.339
PZ_CC = 1.397
PZ_CNC_DEG = 115.6
R_CH = 1.08


def pyrazine_ring():
    """Ring atoms of pyrazine in a local frame with N1 at origin and N4 on +z.

    Returns list of (element, (x, z)) and the N...N span.
    """
    half = np.radians(PZ_CNC_DEG / 2)
    x = PZ_CN * np.sin(half)
    z1 = PZ_CN * np.cos(half)
    span = 2 * z1 + PZ_CC
    atoms = [("N", (0.0, 0.0)), ("N", (0.0, span)),
             ("C", (x, z1)), ("C", (-x, z1)), ("C", (x, span - z1)), ("C", (-x, span - z1))]
    # H on C, along the bisector of the external angle (approximately radial from ring centre)
    cz = span / 2
    for el, (ax, az) in list(atoms):
        if el == "C":
            v = np.array([ax, az - cz])
            v /= np.linalg.norm(v)
            atoms.append(("H", (ax + R_CH * v[0], az + R_CH * v[1])))
    return atoms, span


def octahedron_volume(r_ax1, r_ax2, r_eq):
    """Volume of an octahedron with equatorial bond r_eq and axial bonds r_ax1, r_ax2."""
    return 2 * r_eq ** 2 * (r_ax1 + r_ax2) / 3.0


def coordination_sphere(state="LS"):
    r = R_FEN[state]
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return dirs * r


def hofmann_cell(state="LS"):
    r = R_FEN[state]
    fe_pt = r + R_CN + R_PTC
    a = np.sqrt(2.0) * fe_pt
    _, span = pyrazine_ring()
    c = 2 * r + span
    return dict(a=a, c=c, V=a * a * c, fe_pt=fe_pt, pz_span=span)


def hofmann_atoms(state="LS"):
    """Cartesian atoms (element, xyz) of one tetragonal cell, Fe at origin, Pt at (a/2, a/2, 0).

    Cyanides bridge Pt to the four Fe corners along the in-plane diagonals;
    pyrazine pillars along c, ring plane taken as the (110) diagonal plane.
    """
    cell = hofmann_cell(state)
    a, c = cell["a"], cell["c"]
    r = R_FEN[state]
    atoms = [("Fe", np.array([0.0, 0.0, 0.0])), ("Pt", np.array([a / 2, a / 2, 0.0]))]
    pt = np.array([a / 2, a / 2, 0.0])
    for sx in (-1, 1):
        for sy in (-1, 1):
            u = np.array([sx, sy, 0.0]) / np.sqrt(2)
            atoms.append(("C", pt + R_PTC * u))
            atoms.append(("N", pt + (R_PTC + R_CN) * u))
    ring, span = pyrazine_ring()
    inplane = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)  # ring plane contains c and this
    for el, (x, z) in ring:
        atoms.append((el, inplane * x + np.array([0, 0, r + z])))
    # wrap into the cell
    out = []
    L = np.array([a, a, c])
    for el, p in atoms:
        out.append((el, np.mod(p, L)))
    return out, cell


def summary():
    ls, hs = hofmann_cell("LS"), hofmann_cell("HS")
    v_ls = octahedron_volume(R_FEN["LS"], R_FEN["LS"], R_FEN["LS"])
    v_hs = octahedron_volume(R_FEN["HS"], R_FEN["HS"], R_FEN["HS"])
    fe_used = 2.6  # per nm^3, the readout budget's assumed iron density (paper)
    return dict(
        R_LS=R_FEN["LS"], R_HS=R_FEN["HS"], dR=R_FEN["HS"] - R_FEN["LS"],
        dvol_poly_pct=100 * (v_hs / v_ls - 1),
        a_LS=ls["a"], a_HS=hs["a"], c_LS=ls["c"], c_HS=hs["c"],
        V_LS=ls["V"], V_HS=hs["V"],
        da_pct=100 * (hs["a"] / ls["a"] - 1), dc_pct=100 * (hs["c"] / ls["c"] - 1),
        dV_pct=100 * (hs["V"] / ls["V"] - 1),
        Fe_per_nm3_LS=1000 / ls["V"], Fe_per_nm3_HS=1000 / hs["V"],
        Fe_per_nm3_used=fe_used, V_used=1000 / fe_used,
        pz_span=ls["pz_span"],
    )
