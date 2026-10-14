"""Design condition (manuscript eq. 1-2) and the classical surrogate hierarchy."""
import numpy as np
from scipy import stats

from . import constants as C
from . import kanamori as K


def t_half(dH_cm, dS=C.DS_DEFAULT):
    """Switching temperature T = dH/dS (no clipping; dH<=0 gives T<=0 = high spin always)."""
    return np.asarray(dH_cm, float) * C.CM_TO_JMOL / dS


def k_per_cm(dS=C.DS_DEFAULT):
    return C.CM_TO_JMOL / dS


def success(T_rest, T_ana, t_lo=C.T_LO, t_hi=C.T_HI):
    lo = np.minimum(T_rest, T_ana)
    hi = np.maximum(T_rest, T_ana)
    return (lo <= t_lo) & (hi >= t_hi)


def pair_success(space, T):
    p = space.pairs
    return success(T[p[:, 0]], T[p[:, 1]])


def three_state(T, t_lo=C.T_LO, t_hi=C.T_HI):
    return np.where(T <= t_lo, 0, np.where(T >= t_hi, 2, 1))


class Classical:
    """Exact, mean-field (depth 0) and PT2 (depth 1) energetics for a Space."""

    def __init__(self, space, dS=C.DS_DEFAULT):
        self.space = space
        self.dS = dS
        D, J = space.Delta, space.J
        self.E_ls = K.exact_ls(D, J)
        self.E_ref = K.e_ref(D, J)
        self.E_hs = K.e_hs(D, J)
        self.E2 = K.e_pt2(D, J)
        self.Ecorr = self.E_ls - self.E_ref
        self.dH_exact = self.E_hs - self.E_ls
        self.dH_mf = K.dH_mf(D, J)
        self.dH_pt = self.dH_mf - self.E2
        self.T_exact = t_half(self.dH_exact, dS)
        self.T_mf = t_half(self.dH_mf, dS)
        self.T_pt = t_half(self.dH_pt, dS)
        self.ok_exact = pair_success(space, self.T_exact)


def recalibrate(E2_cal, Ecorr_cal):
    """Least-squares E_corr ~ a + b E2 ; return (a, b, residual sd, design matrix stats)."""
    X = np.column_stack([np.ones_like(E2_cal), E2_cal])
    coef, *_ = np.linalg.lstsq(X, Ecorr_cal, rcond=None)
    resid = Ecorr_cal - X @ coef
    n = len(E2_cal)
    s2 = np.sum(resid ** 2) / max(n - 2, 1)
    return coef[0], coef[1], np.sqrt(s2), X


def fit_se(E2_cal, s_resid, E2_eval):
    """Standard error of the fitted mean at E2_eval from residual variance and leverage."""
    n = len(E2_cal)
    xm = E2_cal.mean()
    sxx = np.sum((E2_cal - xm) ** 2)
    return s_resid * np.sqrt(1.0 / n + (E2_eval - xm) ** 2 / sxx)


def kendall(a, b):
    return float(stats.kendalltau(a, b).statistic)
