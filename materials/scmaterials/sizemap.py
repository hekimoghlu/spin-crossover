"""Slichter-Drickamer mean-field hysteresis of a spin-crossover particle vs size.

Gibbs energy per mole of switching centres (x = high-spin fraction):
    G(x) = x dH - T x dS + Gamma x (1 - x) + R T [x ln x + (1-x) ln(1-x)]
Equilibrium branches: dH - T dS + Gamma (1 - 2x) + R T ln(x/(1-x)) = 0, i.e.
    T(x) = (dH + Gamma (1 - 2x)) / (dS - R ln(x/(1-x))).
The loop is bounded by the turning points (spinodals) of T(x).  With an
inactive surface layer delta, Gamma(d) = Gamma_bulk (1 - 2 delta/d)^3.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

R_GAS = 8.314462618  # J mol^-1 K^-1


def gamma_of_d(d_nm, gamma_bulk, delta_nm=1.0):
    d = np.asarray(d_nm, dtype=float)
    frac = np.clip(1.0 - 2.0 * delta_nm / d, 0.0, None)
    return gamma_bulk * frac ** 3


def d_critical(gamma_bulk, T_half=310.0, delta_nm=1.0):
    """Diameter at which Gamma(d) = 2 R T_half (loop closes)."""
    r = 2 * R_GAS * T_half / gamma_bulk
    if r >= 1:
        return np.inf
    return 2 * delta_nm / (1 - r ** (1.0 / 3.0))


def T_of_x(x, gamma, T_half=310.0, dS=60.0):
    dH = T_half * dS
    return (dH + gamma * (1 - 2 * x)) / (dS - R_GAS * np.log(x / (1 - x)))


def loop_width(gamma, T_half=310.0, dS=60.0):
    """Width T_up - T_down (K) of the mean-field hysteresis loop; 0 if no loop.

    T_up (heating, LS->HS jump) is the local maximum of T(x) on the low-x side,
    T_down (cooling) the local minimum on the high-x side.
    """
    if gamma <= 2 * R_GAS * T_half:  # no bistability (exact: at x=1/2, T=T_half)
        return 0.0
    f = lambda x: T_of_x(x, gamma, T_half, dS)
    # turning points are symmetric-ish around 1/2; search on each side
    eps = 1e-12
    r1 = minimize_scalar(lambda x: -f(x), bounds=(eps, 0.5), method="bounded",
                         options={"xatol": 1e-13})
    r2 = minimize_scalar(f, bounds=(0.5, 1 - eps), method="bounded",
                         options={"xatol": 1e-13})
    Tup, Tdn = -r1.fun, r2.fun
    return max(0.0, float(Tup - Tdn))


def loop_branches(gamma, T_half=310.0, dS=60.0, n=2001):
    """Heating and cooling branches x_HS(T) of the mean-field loop (for plotting)."""
    x = np.linspace(1e-6, 1 - 1e-6, n)
    T = T_of_x(x, gamma, T_half, dS)
    # heating: follow low-x branch until turning point, then jump
    i_up = np.argmax(np.where(x < 0.5, T, -np.inf)) if gamma > 2 * R_GAS * T_half else None
    i_dn = np.argmin(np.where(x > 0.5, T, np.inf)) if gamma > 2 * R_GAS * T_half else None
    return x, T, i_up, i_dn


def size_table(diameters=(10, 20, 50, 100, 200), gammas=(6e3, 8e3, 10e3),
               delta_nm=1.0, T_half=310.0, dS=60.0):
    out = {}
    for g in gammas:
        row = {str(d): loop_width(float(gamma_of_d(d, g, delta_nm)), T_half, dS) for d in diameters}
        row["bulk"] = loop_width(g, T_half, dS)
        row["d_c"] = d_critical(g, T_half, delta_nm)
        out[f"{g/1e3:g}"] = row
    return out


def mf_hysteresis_sweep(gamma, T_half=310.0, dS=60.0, Tmin=200.0, Tmax=420.0, nT=441):
    """Deterministic quasi-static sweep: follow the stable root continuously.

    Returns T grid, x_heating, x_cooling.  Used to cross-check the Rust code.
    """
    Ts = np.linspace(Tmin, Tmax, nT)
    dH = T_half * dS

    def F(x, T):
        return dH - T * dS + gamma * (1 - 2 * x) + R_GAS * T * np.log(x / (1 - x))

    def follow(Tseq, x0):
        xs = []
        x = x0
        for T in Tseq:
            # roots of F in (0,1): bracket scan, pick the one nearest the previous x
            grid = np.linspace(1e-12, 1 - 1e-12, 4001)
            v = F(grid, T)
            roots = []
            for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
                roots.append(brentq(F, grid[i], grid[i + 1], args=(T,), xtol=1e-14))
            # stable roots have dF/dx > 0
            stable = [r for r in roots if (gamma * -2 + R_GAS * T / (r * (1 - r))) > 0]
            x = min(stable, key=lambda r: abs(r - x))
            xs.append(x)
        return np.array(xs)

    xh = follow(Ts, 0.0)
    xc = follow(Ts[::-1], 1.0)[::-1]
    return Ts, xh, xc
