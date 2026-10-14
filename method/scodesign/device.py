"""Simulated device evaluations: ansatz + twin circuits, drift epochs, shot levels, four arms.

For each (complex, epoch) the noisy density matrices at the three folding scales are computed
once; every shot level then draws independent multinomial histograms from them:
  raw arm        : 3 histograms (scale 1, Z/X/Y), uncorrected
  mitigated arm  : 9 histograms (3 scales x 3 bases), tensored readout correction + ZNE
  twin circuits  : same, with all Givens angles 0 (reference-state twin, sec. 6.1)
Derived arms: reference-scaled  E_raw * E_ref / E_ref,raw ;
              paired-mitigated  E_mit - (E_ref,mit - E_ref).
"""
import numpy as np

from . import ansatz as A
from . import constants as C
from . import kanamori as K
from . import noise as N

SCALES = C.ZNE_SCALES


class Program:
    _cache = {}

    @classmethod
    def get(cls, routed, scale):
        key = (routed, scale)
        if key not in cls._cache:
            code, coef = A.noisy_program(routed, scale)
            perm = A.wire_to_logical_perm(routed)
            cls._cache[key] = (code, coef, A.initial_index(routed), perm)
        return cls._cache[key]


def _probs(routed, scale, params, ro, backend="auto", chunk=4096):
    code, coef, init, perm = Program.get(routed, scale)
    out = []
    for i in range(0, params.shape[0], chunk):
        rho = N.run_batch(code, coef, params[i:i + chunk], init, backend=backend)
        r = np.empty_like(rho)
        r[:, perm[:, None], perm[None, :]] = rho   # wire order -> logical qubit order
        P = N.basis_probs(r)
        out.append(N.apply_readout(P, ro))
    return np.concatenate(out, 0)


def evaluate(Delta, J, thetas, platform, z, shots_list, rng, twin=True, backend="auto"):
    """Run evaluations.

    Delta, J: (nc,) ; thetas (nc,6) ; platform dict with r2q, r1q, ro, routed ;
    z: (nc, ne) standard normal drift draws.
    Returns dict of arrays with shape (nc, ne, n_shots).
    """
    nc, ne = z.shape
    lam1 = 2.0 * platform["r1q"]
    lam2 = 4.0 / 3.0 * platform["r2q"] * np.exp(C.DRIFT_SIGMA * z)  # (nc, ne)
    routed = platform["routed"]
    ro = platform["ro"]
    circuits = [("ans", thetas)] + ([("twin", np.zeros_like(thetas))] if twin else [])
    coeffs = K.pauli_coeff_arrays(Delta, J)
    co_rep = tuple(np.repeat(c, ne, axis=0) for c in coeffs)
    res = {}
    for name, th in circuits:
        params = np.zeros((nc * ne, A.NPARAM))
        params[:, :6] = np.repeat(th, ne, axis=0)
        params[:, A.P_L1] = lam1
        params[:, A.P_L2] = lam2.reshape(-1)
        Pm = np.stack([_probs(routed, s, params, ro, backend) for s in SCALES], 1)  # (nc*ne, 3 scales, 3 bases, 32)
        # exact (infinite-shot) noisy energies at scale 1, for diagnostics
        E_inf = N.energy_from_freqs(N.correct_readout(Pm[:, 0], ro), co_rep)
        Eraw = np.empty((nc * ne, len(shots_list)))
        Emit = np.empty_like(Eraw)
        Eexp = np.zeros_like(Eraw, dtype=bool)
        Escl = np.empty((nc * ne, len(shots_list), len(SCALES)))
        for si, shots in enumerate(shots_list):
            Fraw = N.sample(rng, Pm[:, 0], shots)                      # independent raw draw
            Eraw[:, si] = N.energy_from_freqs(Fraw, co_rep)
            Fm = N.sample(rng, Pm, shots)                               # 9 histograms
            Fc = N.correct_readout(Fm, ro)
            Es = np.stack([N.energy_from_freqs(Fc[:, k], co_rep) for k in range(len(SCALES))], 1)
            Escl[:, si, :] = Es
            e0, ex = N.zne(Es, SCALES, c0=co_rep[3])
            Emit[:, si] = e0
            Eexp[:, si] = ex
        res[name] = dict(raw=Eraw.reshape(nc, ne, -1), mit=Emit.reshape(nc, ne, -1),
                         exp=Eexp.reshape(nc, ne, -1), inf=E_inf.reshape(nc, ne),
                         scales=Escl.reshape(nc, ne, len(shots_list), len(SCALES)))
    out = dict(raw=res["ans"]["raw"], mit=res["ans"]["mit"], exp=res["ans"]["exp"], inf=res["ans"]["inf"],
               scales=res["ans"]["scales"])   # readout-corrected energies at the three folding scales
    if twin:
        Eref = K.e_ref(Delta, J)[:, None, None]
        out["ref_raw"] = res["twin"]["raw"]
        out["ref_mit"] = res["twin"]["mit"]
        out["ref_scales"] = res["twin"]["scales"]
        out["rsc"] = out["raw"] * Eref / out["ref_raw"]
        out["pmit"] = out["mit"] - (out["ref_mit"] - Eref)
    return out


def drift_draws(seed_tuple, nc, ne):
    rng = np.random.default_rng(seed_tuple)
    return rng.standard_normal((nc, ne))
