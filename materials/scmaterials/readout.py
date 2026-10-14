"""Readout budgets: NV magnetometry, array pitch, response time, graphene interface."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

MU_B = 9.2740100783e-24     # J/T
K_B = 1.380649e-23          # J/K
MU0_4PI = 1e-7              # T m / A
E_CH = 1.602176634e-19
HBAR = 1.054571817e-34
EPS0 = 8.8541878128e-12
DEBYE = 3.33564e-30         # C m
V_F_GRAPHENE = 1.0e6        # m/s (assumed standard value)


# --------------------------------------------------------------------------
# magnetic readout
# --------------------------------------------------------------------------
def curie_moment_per_fe(T=310.0, B=0.1, S=2, g=2.0):
    """Thermal-average moment (in mu_B) of a Curie paramagnet, linear regime."""
    return g * g * S * (S + 1) * MU_B * B / (3 * K_B * T)


def n_fe(d_nm, rho_fe_nm3=2.6):
    return np.pi / 6 * np.asarray(d_nm, float) ** 3 * rho_fe_nm3


def stray_field_contrast(d_nm, standoff_nm, T=310.0, B=0.1, rho_fe_nm3=2.6):
    """On-axis point-dipole field (T) of the HS-LS moment difference at a sensor
    standoff_nm from the particle surface."""
    m = n_fe(d_nm, rho_fe_nm3) * curie_moment_per_fe(T, B) * MU_B
    r = (np.asarray(standoff_nm, float) + np.asarray(d_nm, float) / 2) * 1e-9
    return MU0_4PI * 2 * m / r ** 3


def time_to_snr(field_T, eta_T_rtHz, snr=3.0):
    """Averaging time for SNR = field * sqrt(t) / eta."""
    return (snr * eta_T_rtHz / np.asarray(field_T, float)) ** 2


def neighbour_ratio(pitch_nm, height_nm, nmax=400):
    """|sum of z-fields of all other particles| / own field, at a sensor height
    above the central particle of a square array of identical z-dipoles."""
    i = np.arange(-nmax, nmax + 1)
    X, Y = np.meshgrid(i, i, indexing="ij")
    rho2 = (X ** 2 + Y ** 2).astype(float) * pitch_nm ** 2
    h = float(height_nm)
    r2 = rho2 + h * h
    bz = (3 * h * h / r2 ** 2.5 - 1 / r2 ** 1.5)
    bz[nmax, nmax] = 0.0
    own = 2 / h ** 3
    return abs(bz.sum()) / own


def critical_pitch(height_nm, threshold=0.1, pmax_factor=20.0):
    """Smallest pitch above which the neighbour field stays below threshold."""
    ps = np.linspace(0.3 * height_nm, pmax_factor * height_nm, 240)
    rat = np.array([neighbour_ratio(p, height_nm) for p in ps])
    above = np.nonzero(rat >= threshold)[0]
    if len(above) == 0:
        return ps[0]
    j = above.max()
    f = lambda p: neighbour_ratio(p, height_nm) - threshold
    return brentq(f, ps[j], ps[j + 1], xtol=1e-6)


# --------------------------------------------------------------------------
# response time
# --------------------------------------------------------------------------
def shell_delay(L_m, D_m2s):
    return L_m ** 2 / (2 * D_m2s)


def response_budget(k_ex=4.4e6, K_os=1.0):
    return dict(
        tau_shell_gel_s=shell_delay(10e-9, 1e-10),
        tau_shell_silica_s=shell_delay(50e-9, 1e-12),
        tau_exchange_s=1 / k_ex,
        tau_assoc_20uM_s=1 / (K_os * k_ex * 20e-6),
        tau_assoc_100nM_s=1 / (K_os * k_ex * 100e-9),
    )


# --------------------------------------------------------------------------
# graphene interface
# --------------------------------------------------------------------------
def graphene_budget(dp_debye=1.0, sigma_per_nm2=1.0, eps_r=3.0):
    sigma = sigma_per_nm2 * 1e18
    dphi = sigma * dp_debye * DEBYE / (EPS0 * eps_r)  # V
    EF = E_CH * dphi
    kF = EF / (HBAR * V_F_GRAPHENE)
    n = kF ** 2 / np.pi  # m^-2 (g_s g_v = 4: n = kF^2/pi)
    n_cm2 = n * 1e-4
    return dict(dphi_mV=dphi * 1e3, dn_cm2=n_cm2,
                rel_to_1e12_pct=100 * n_cm2 / 1e12, rel_to_1e13_pct=100 * n_cm2 / 1e13)


def nv_budget():
    m = curie_moment_per_fe()
    b50_100 = stray_field_contrast(50, 100)
    b100_50 = stray_field_contrast(100, 50)
    table = {d: {s: stray_field_contrast(d, s) * 1e9 for s in (10, 50, 100, 200)} for d in (20, 50, 100, 200)}
    return dict(
        moment_per_fe_muB=m,
        n_fe_50nm=float(n_fe(50)),
        field_50nm_at_100nm_nT=b50_100 * 1e9,
        field_100nm_at_50nm_uT=b100_50 * 1e6,
        t_typ_50_100_s=time_to_snr(b50_100, 1e-6),
        t_best_50_100_s=time_to_snr(b50_100, 10e-9),
        t_typ_100_50_s=time_to_snr(b100_50, 1e-6),
        table_nT=table,
    )
