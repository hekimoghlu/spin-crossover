"""End-to-end design loop (manuscript sections 2-5 and the reimplemented parts of section 6).

All outputs are computed here; the manuscript's reported numbers are never read by this module.
`results["macros"]` holds this reimplementation's values keyed by the manuscript's macro names
so that `validation.py` can line them up against the paper.
"""
import json
import os
import time
import zlib

import numpy as np
from scipy.optimize import brentq

from . import ansatz as A
from . import constants as C
from . import design
from . import device as Dv
from . import kanamori as K
from . import ligands as L
from . import pricing

PLAT = list(C.PLATFORMS.keys())
ROUTABLE_USD = ["Cepheus", "Heron", "Forte"]
ROUTABLE_HQC = ["Helios"]


def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


class Config:
    def __init__(self, n_epochs=C.N_EPOCHS, n_epochs_cal=C.N_EPOCHS_CALCOMP, shots=tuple(C.SHOTS),
                 seed=C.SEED, backend="auto", rsem_epochs=6, budget_subsets=30, rough_reps=24,
                 rough_seeds=3, eps_epochs=C.EPS_EPOCHS, quick=False, rdm_steps=3000, rl_episodes=40000):
        self.n_epochs = n_epochs
        self.n_epochs_cal = n_epochs_cal
        self.shots = list(shots)
        self.seed = seed
        self.backend = backend
        self.rsem_epochs = min(rsem_epochs, n_epochs)
        self.budget_subsets = budget_subsets
        self.rough_reps = rough_reps
        self.rough_seeds = rough_seeds
        self.eps_epochs = eps_epochs
        self.quick = quick
        self.rdm_steps = rdm_steps
        self.rl_episodes = rl_episodes

    def as_dict(self):
        return dict(self.__dict__)


class Loop:
    def __init__(self, cfg=None, log=print):
        self.cfg = cfg or Config()
        self.log = log
        self.R = {"config": self.cfg.as_dict(), "macros": {}, "timing": {}}
        self.M = self.R["macros"]

    # ---------------------------------------------------------------- helpers
    def rng(self, *tag):
        # deterministic (crc32, not Python's salted str hash) so runs are reproducible
        return np.random.default_rng([self.cfg.seed] + [zlib.crc32(t.encode()) if isinstance(t, str) else int(t)
                                                        for t in tag])

    def kpc(self):
        return design.k_per_cm()

    def T_from_Els(self, idx, E_ls):
        """Temperature from a low-spin energy estimate for complexes idx."""
        return design.t_half(self.cl.E_hs[idx] - E_ls)

    # ---------------------------------------------------------------- stage 1: classical
    def classical(self):
        t0 = time.time()
        sp = self.sp = L.Space()
        cl = self.cl = design.Classical(sp)
        M = self.M
        M["nScaffolds"] = len(sp.scaffolds)
        M["nComplexes"] = sp.n_complexes
        M["nPairs"] = sp.n_pairs
        M["lsFraction"] = 100 * float(np.mean(cl.dH_exact > 0))
        ok_mf = design.pair_success(sp, cl.T_mf)
        ok_pt = design.pair_success(sp, cl.T_pt)
        M["nOkExact"] = int(cl.ok_exact.sum())
        M["nOkMF"] = int(ok_mf.sum())
        M["nOkPT"] = int(ok_pt.sum())
        M["agreeMF"] = 100 * float(np.mean(ok_mf == cl.ok_exact))
        M["agreePT"] = 100 * float(np.mean(ok_pt == cl.ok_exact))
        e = cl.T_mf - cl.T_exact
        M["rmsMF"], M["biasMF"] = _rms(e), float(e.mean())
        e = cl.T_pt - cl.T_exact
        M["rmsPT"], M["biasPT"], M["maxPT"] = _rms(e), float(e.mean()), float(np.abs(e).max())
        M["PoverB"] = 2.5 + 4 * C.C_OVER_B
        ec = -cl.Ecorr
        M["ecorrMin"], M["ecorrMax"] = float(ec.min()), float(ec.max())
        M["ecorrMinK"], M["ecorrMaxK"] = float(ec.min() * self.kpc()), float(ec.max() * self.kpc())
        M["deltaMin"], M["deltaMax"] = float(sp.Delta.min()), float(sp.Delta.max())
        M["Bmin"], M["Bmax"] = float(sp.B.min()), float(sp.B.max())
        M["dHlo"] = C.T_LO / self.kpc()
        M["dHhi"] = C.T_HI / self.kpc()
        M["dHmarginMHa"] = C.MARGIN / self.kpc() / C.HARTREE_CM * 1000
        M["tolFloorCm"] = C.TAU_FLOOR / self.kpc()
        names = list(C.SIXTH_SITE)
        for xi, key in [(1, "nOkCl"), (2, "nOkNthree"), (3, "nOkNCS"), (4, "nOkCN")]:
            M[key] = int(cl.ok_exact[sp.pairs[:, 3] == xi].sum())
        p = sp.pairs
        tr, ta = cl.T_exact[p[:, 0]], cl.T_exact[p[:, 1]]
        M["nDesignsHStoLS"] = int(np.sum(cl.ok_exact & (tr <= C.T_LO) & (ta >= C.T_HI)))
        M["nDesignsLStoHS"] = int(np.sum(cl.ok_exact & (tr >= C.T_HI) & (ta <= C.T_LO)))
        self.R["max_resting_T"] = float(cl.T_exact[sp.sixth_idx == 0].max())
        # calibration compound
        D, B = L.ligand_field(C.CAL_COMPOUND)
        J = L.pairing_energy(B) / 4
        dh = float(K.e_hs(D, J) - K.exact_ls(D, J)[0])
        M["calDelta"], M["calB"], M["caldH"], M["calT"] = D, B, dh, float(design.t_half(dh))
        self.calc = dict(Delta=D, B=B, J=J)
        # model sweep at B = 700 (fig 1a)
        Bf = 700.0
        Jf = L.pairing_energy(Bf) / 4
        gap_ex = lambda x: float(K.e_hs(x * Bf, Jf) - K.exact_ls(x * Bf, Jf)[0])
        gap_mf = lambda x: float(K.dH_mf(x * Bf, Jf))
        gap_pt = lambda x: float(K.dH_mf(x * Bf, Jf) - K.e_pt2(x * Bf, Jf)[0])
        M["xoExact"] = brentq(gap_ex, 10, 30)
        M["xoMF"] = brentq(gap_mf, 10, 30)
        M["xoPT"] = brentq(gap_pt, 10, 30)
        xs = np.linspace(12, 24, 121)
        self.R["sweep"] = dict(x=xs.tolist(), exact=[gap_ex(x) for x in xs], mf=[gap_mf(x) for x in xs],
                               pt=[gap_pt(x) for x in xs])
        # full-space vs block cross-check on twelve random complexes
        rng = self.rng("fullspace")
        pick = rng.choice(sp.n_complexes, 12, replace=False)
        dev_s, dev_q, comm = [], [], []
        dev_s_std = []
        for i in pick:
            chk = K.full_space_check(sp.Delta[i], sp.J[i], spin_flip_sign=+1.0)
            chk2 = K.full_space_check(sp.Delta[i], sp.J[i], spin_flip_sign=-1.0)
            dev_s.append(abs(chk["e_singlet"] - cl.E_ls[i]))
            dev_s_std.append(abs(chk2["e_singlet"] - cl.E_ls[i]))
            dev_q.append(abs(chk["e_quintet"] - cl.E_hs[i]))
            comm.append(chk["commutator_H_S2"])
        self.R["fullspace_check"] = dict(complexes=pick.tolist(), max_dev_singlet=float(max(dev_s)),
                                         max_dev_singlet_standard_sign=float(max(dev_s_std)),
                                         max_dev_quintet=float(max(dev_q)), max_comm=float(max(comm)))
        # Pauli structure
        c0, terms = K.pauli_terms(sp.Delta[0], sp.J[0])
        groups = K.measurement_groups(terms)
        self.R["pauli"] = dict(n_terms=len(terms), c0=c0, groups={k: len(v) for k, v in groups.items()})
        self.R["timing"]["classical"] = time.time() - t0
        self.log(f"[classical] {M['nComplexes']} complexes, {M['nOkExact']} successful designs, "
                 f"LS fraction {M['lsFraction']:.1f}%")

    # ---------------------------------------------------------------- angles
    def angles(self, idx):
        th = np.zeros((len(idx), 6))
        en = np.zeros(len(idx))
        for k, i in enumerate(idx):
            if i in self._angle_cache:
                th[k], en[k] = self._angle_cache[i]
                continue
            H = K.hamiltonian_matrix(self.sp.Delta[i], self.sp.J[i])
            t, e = A.optimise_angles(H, n_starts=3, seed=int(i))
            self._angle_cache[i] = (t, e)
            th[k], en[k] = t, e
        return th, en

    # ---------------------------------------------------------------- device evaluation wrapper
    def run_device(self, Delta, J, thetas, n_epochs, tag, platforms=None, twin=True):
        out = {}
        for pi, pname in enumerate(platforms or PLAT):
            p = C.PLATFORMS[pname]
            z = Dv.drift_draws([self.cfg.seed, 7, pi] + [ord(c) for c in tag], len(Delta), n_epochs)
            rng = self.rng("shots", tag, pname)
            out[pname] = Dv.evaluate(np.asarray(Delta), np.asarray(J), thetas, p, z, self.cfg.shots, rng,
                                     twin=twin, backend=self.cfg.backend)
        return out

    # ---------------------------------------------------------------- stage 2: router predictions
    def router_predictions(self):
        t0 = time.time()
        c = self.calc
        H = K.hamiltonian_matrix(c["Delta"], c["J"])
        th, _ = A.optimise_angles(H, n_starts=3, seed=1)
        ev = self.run_device(np.array([c["Delta"]]), np.array([c["J"]]), th[None], self.cfg.n_epochs_cal, "calcomp")
        Ex = K.exact_ls(c["Delta"], c["J"])[0]
        pred = {}
        for pname in PLAT:
            pred[pname] = {}
            for arm in ("raw", "mit", "rsc", "pmit"):
                err = -(ev[pname][arm][0] - Ex) * self.kpc()   # (epochs, shots)
                pred[pname][arm] = dict(rmse=[_rms(err[:, s]) for s in range(len(self.cfg.shots))],
                                        bias=[float(err[:, s].mean()) for s in range(len(self.cfg.shots))])
        self.pred = pred
        self.R["pred"] = pred
        for pname in PLAT:
            for si, tag in enumerate(C.SHOT_TAGS[:len(self.cfg.shots)]):
                for arm, an in (("raw", "Raw"), ("mit", "Mit")):
                    self.M[f"predRmse{pname}{an}{tag}"] = pred[pname][arm]["rmse"][si]
                    self.M[f"predBias{pname}{an}{tag}"] = pred[pname][arm]["bias"][si]
                    cst = pricing.eval_cost(pname, arm, self.cfg.shots[si])
                    if cst is not None:
                        self.M[f"predMoney{pname}{an}{tag}"] = cst
        self.R["timing"]["router_predictions"] = time.time() - t0

    def choose_arm(self, tol, arms=("raw", "mit")):
        """Cheapest USD arm with predicted error <= tol; else cheapest HQC arm; else most accurate priced arm."""
        cands = []
        for pname in ROUTABLE_USD + ROUTABLE_HQC:
            for arm in arms:
                for si, shots in enumerate(self.cfg.shots):
                    cands.append((pname, arm, si, self.pred[pname][arm]["rmse"][si],
                                  pricing.eval_cost(pname, arm, shots), pricing.currency(pname)))
        usd = [c for c in cands if c[5] == "USD" and c[3] <= tol]
        if usd:
            return min(usd, key=lambda c: (c[4], c[3]))[:3]
        hqc = [c for c in cands if c[5] == "HQC" and c[3] <= tol]
        if hqc:
            return min(hqc, key=lambda c: (c[4], c[3]))[:3]
        return min(cands, key=lambda c: c[3])[:3]

    # ---------------------------------------------------------------- stage 3: calibration + dispatch
    def calibrate(self):
        t0 = time.time()
        sp, cl, M = self.sp, self.cl, self.M
        rng = self.rng("calset")
        self.cal = np.sort(rng.choice(sp.n_complexes, C.N_CAL, replace=False))
        tol = C.S_TARGET * np.sqrt(C.N_CAL)
        M["nCal"] = C.N_CAL
        M["calTol"] = tol
        M["sTarget"] = C.S_TARGET
        self.cal_arm = self.choose_arm(tol)
        pname, arm, si = self.cal_arm
        self.R["cal_arm"] = dict(platform=pname, arm=arm, shots=self.cfg.shots[si])
        M["calArm"] = f"{C.PLATFORMS[pname]['label']}, {'mitigated' if arm == 'mit' else 'raw'} arm, {self.cfg.shots[si]:,} shots per basis"
        th, en = self.angles(self.cal)
        self.ev_cal = self.run_device(sp.Delta[self.cal], sp.J[self.cal], th, self.cfg.n_epochs, "cal")
        self.ans_energy_cal = en
        # deployed fit: chosen arm, epoch 0
        E_dev = self.ev_cal[pname][arm][:, 0, si]
        fit = self.fit_surrogate(E_dev)
        self.fit = fit
        self.T2 = fit["T"]
        heldout = np.setdiff1d(np.arange(sp.n_complexes), self.cal)
        self.heldout = heldout
        e = self.T2[heldout] - cl.T_exact[heldout]
        M["rmsPTc"], M["biasPTc"], M["maxPTc"] = _rms(e), float(e.mean()), float(np.abs(e).max())
        M["corrA"], M["corrB"] = fit["a"], fit["b"]
        M["residPTc"] = fit["resid_sd"] * self.kpc()
        M["fitsePTc"] = fit["se_mean_K"]
        M["band"] = C.BAND_FACTOR * fit["se_mean_K"]
        ok2 = design.pair_success(sp, self.T2)
        M["nOkPTc"] = int(ok2.sum())
        M["agreePTc"] = 100 * float(np.mean(ok2 == cl.ok_exact))
        # device scatter of calibration points about the exact correlation energies
        Ecorr_dev = E_dev - cl.E_ref[self.cal]
        self.R["cal_scatter_cm"] = _rms(Ecorr_dev - cl.Ecorr[self.cal])
        self.R["cal_points"] = dict(E2=cl.E2[self.cal].tolist(), Ecorr_dev=Ecorr_dev.tolist())
        # dispatch
        band = M["band"]
        dist = np.minimum(np.abs(self.T2 - C.T_LO), np.abs(self.T2 - C.T_HI))
        mask = dist <= band
        mask[self.cal] = False
        self.disp = np.where(mask)[0]
        M["nDispatched"] = int(len(self.disp))
        dpairs = np.where(np.isin(sp.pairs[:, 1], self.disp))[0]
        self.dpairs = dpairs
        M["nDispPairs"] = int(len(dpairs))
        Tex = cl.T_exact[sp.pairs[dpairs, 1]]
        hard = (np.abs(Tex - C.T_LO) <= 50) | (np.abs(Tex - C.T_HI) <= 50)
        self.hard = hard
        M["nHardPairs"] = int(hard.sum())
        M["nDispLS"] = int(np.sum(cl.dH_exact[self.disp] > 0))
        M["nEvalComplexes"] = int(len(self.cal) + len(self.disp))
        M["epochs"] = self.cfg.n_epochs
        self.R["timing"]["calibrate"] = time.time() - t0
        self.log(f"[calibrate] arm {M['calArm']}; held-out RMS {M['rmsPTc']:.1f} K; band {band:.1f} K; "
                 f"dispatched {M['nDispatched']}")

    def fit_surrogate(self, E_dev_cal, cal=None, Ecorr_offset=None):
        """Depth-2 recalibration from device low-spin energies on the calibration set."""
        cal = self.cal if cal is None else cal
        cl = self.cl
        Ecorr_dev = E_dev_cal - cl.E_ref[cal]
        a, b, s, _ = design.recalibrate(cl.E2[cal], Ecorr_dev)
        E_ls2 = cl.E_ref + a + b * cl.E2
        T = design.t_half(cl.E_hs - E_ls2)
        se = design.fit_se(cl.E2[cal], s, cl.E2) * self.kpc()
        return dict(a=float(a), b=float(b), resid_sd=float(s), T=T, se_mean_K=float(se.mean()), E_ls=E_ls2)

    # ---------------------------------------------------------------- stage 4: dispatched evaluations
    def dispatched(self):
        t0 = time.time()
        sp = self.sp
        th, en = self.angles(self.disp)
        self.theta_disp = th
        self.ev_disp = self.run_device(sp.Delta[self.disp], sp.J[self.disp], th, self.cfg.n_epochs, "disp")
        allen = np.concatenate([self.ans_energy_cal, en])
        allidx = np.concatenate([self.cal, self.disp])
        resid = allen - self.cl.E_ls[allidx]
        self.M["ansatzResidMax"] = float(resid.max())
        ne, ns = self.cfg.n_epochs, len(self.cfg.shots)
        self.M["nEvals"] = int(len(allidx) * len(PLAT) * ne * ns)
        self.R["timing"]["dispatched"] = time.time() - t0

    def all_errors(self, pname, arm):
        """Temperature errors (n_eval, epochs, shots) over calibration + dispatched complexes."""
        cl = self.cl
        idx = np.concatenate([self.cal, self.disp])
        E = np.concatenate([self.ev_cal[pname][arm], self.ev_disp[pname][arm]], 0)
        return -(E - cl.E_ls[idx][:, None, None]) * self.kpc()

    # ---------------------------------------------------------------- stage 5: tables
    def noise_table(self):
        M = self.M
        tab = []
        for pname in PLAT:
            raw = self.all_errors(pname, "raw")
            mit = self.all_errors(pname, "mit")
            exp = np.concatenate([self.ev_cal[pname]["exp"], self.ev_disp[pname]["exp"]], 0)
            for si, tag in enumerate(C.SHOT_TAGS[:len(self.cfg.shots)]):
                row = dict(platform=pname, shots=self.cfg.shots[si],
                           raw_bias=float(raw[..., si].mean()), raw_rms=_rms(raw[..., si]),
                           mit_bias=float(mit[..., si].mean()), mit_rms=_rms(mit[..., si]),
                           exp_pct=100 * float(exp[..., si].mean()),
                           cost_mit=pricing.eval_cost(pname, "mit", self.cfg.shots[si]),
                           cost_raw=pricing.eval_cost(pname, "raw", self.cfg.shots[si]))
                tab.append(row)
                M[f"rmseRaw{pname}{tag}"] = row["raw_rms"]
                M[f"rmseMit{pname}{tag}"] = row["mit_rms"]
                M[f"biasRawK{pname}{tag}"] = row["raw_bias"]
                M[f"biasMitK{pname}{tag}"] = row["mit_bias"]
                M[f"fracExp{pname}{tag}"] = row["exp_pct"]
                kpc = self.kpc()
                M[f"biasRaw{pname}{tag}"] = row["raw_bias"] / kpc     # same quantities in cm^-1
                M[f"biasMit{pname}{tag}"] = row["mit_bias"] / kpc
                M[f"rmseRawCm{pname}{tag}"] = row["raw_rms"] / kpc
                M[f"rmseMitCm{pname}{tag}"] = row["mit_rms"] / kpc
                if row["cost_mit"] is not None:
                    M[f"moneyMit{pname}{tag}"] = row["cost_mit"]
                    M[f"moneyRaw{pname}{tag}"] = row["cost_raw"]
            M[f"nTwoQ{pname}"] = A.gate_counts(C.PLATFORMS[pname]["routed"], 1.0)[0]
        M["nTwoQNative"] = A.gate_counts(False, 1.0)[0]
        M["nTwoQRouted"] = A.gate_counts(True, 1.0)[0]
        self.R["tab_noise"] = tab

    def surr_table(self):
        tab = []
        for pname in PLAT:
            for arm, an in (("raw", "Raw"), ("mit", "Mit"), ("rsc", "Rsc"), ("pmit", "Pmit")):
                for si, tag in enumerate(C.SHOT_TAGS[:len(self.cfg.shots)]):
                    fit = self.fit_surrogate(self.ev_cal[pname][arm][:, 0, si])
                    e = fit["T"][self.heldout] - self.cl.T_exact[self.heldout]
                    row = dict(platform=pname, arm=arm, shots=self.cfg.shots[si], rms=_rms(e), bias=float(e.mean()),
                               se=fit["se_mean_K"], resid=fit["resid_sd"] * self.kpc())
                    tab.append(row)
                    if arm in ("raw", "mit"):
                        self.M[f"surrRms{pname}{an}{tag}"] = row["rms"]
                        self.M[f"surrBias{pname}{an}{tag}"] = row["bias"]
                        self.M[f"surrSe{pname}{an}{tag}"] = row["se"]
                        self.M[f"surrResid{pname}{an}{tag}"] = row["resid"]
        self.R["tab_surr"] = tab
        mit = [r for r in tab if r["arm"] == "mit"]
        best = min(mit, key=lambda r: r["rms"])
        self.M["surrBestRms"] = best["rms"]
        self.R["surr_best"] = best

    def _policy_row(self, T_disp_est, dH_disp_est):
        """Decision quality on the dispatched pairs; resting complexes from the depth-2 surrogate."""
        sp, cl = self.sp, self.cl
        T = self.T2.copy()
        T[self.disp] = T_disp_est
        ok = design.pair_success(sp, T)[self.dpairs]
        wrong = ok != cl.ok_exact[self.dpairs]
        e = T_disp_est - cl.T_exact[self.disp]
        return dict(wrong=int(wrong.sum()), hard_wrong=int(wrong[self.hard].sum()), rms=_rms(e),
                    tau=design.kendall(dH_disp_est, cl.dH_exact[self.disp]))

    def _classical_row(self, Tall, dHall):
        sp, cl = self.sp, self.cl
        ok = design.pair_success(sp, Tall)[self.dpairs]
        wrong = ok != cl.ok_exact[self.dpairs]
        e = Tall[self.disp] - cl.T_exact[self.disp]
        return dict(wrong=int(wrong.sum()), hard_wrong=int(wrong[self.hard].sum()), rms=_rms(e),
                    tau=design.kendall(dHall[self.disp], cl.dH_exact[self.disp]))

    def policies(self):
        M, cl = self.M, self.cl
        rows = []
        dH2 = self.T2 / self.kpc()
        for name, T, dH, key in (("depth 0, mean field", cl.T_mf, cl.dH_mf, "MF"),
                                 ("depth 1, pair PT2", cl.T_pt, cl.dH_pt, "PT"),
                                 ("depth 2, device-calibrated surrogate", self.T2, dH2, "PTc")):
            r = self._classical_row(T, dH)
            r.update(policy=name, cost=0.0, currency=None)
            rows.append(r)
            M[f"clWrong{key}"], M[f"clHardWrong{key}"], M[f"clRmse{key}"] = r["wrong"], r["hard_wrong"], r["rms"]
        n = len(self.disp)
        for pname in PLAT:
            for arm, an in (("raw", "Raw"), ("mit", "Mit")):
                for si, tag in enumerate(C.SHOT_TAGS[:len(self.cfg.shots)]):
                    E = self.ev_disp[pname][arm][:, 0, si]
                    dH = cl.E_hs[self.disp] - E
                    r = self._policy_row(design.t_half(dH), dH)
                    c = pricing.eval_cost(pname, arm, self.cfg.shots[si])
                    r.update(policy=f"{C.PLATFORMS[pname]['label']}, {arm}, {self.cfg.shots[si]:,}",
                             platform=pname, arm=arm, shots=self.cfg.shots[si],
                             cost=None if c is None else c * n, currency=pricing.currency(pname))
                    rows.append(r)
                    M[f"uWrong{pname}{an}{tag}"] = r["wrong"]
                    M[f"uHardWrong{pname}{an}{tag}"] = r["hard_wrong"]
                    M[f"uRmse{pname}{an}{tag}"] = r["rms"]
                    M[f"uTau{pname}{an}{tag}"] = r["tau"]
                    if r["cost"] is not None:
                        M[f"uCost{pname}{an}{tag}"] = r["cost"]
        # routed policy
        dist = np.minimum(np.abs(self.T2[self.disp] - C.T_LO), np.abs(self.T2[self.disp] - C.T_HI))
        tau = np.maximum(C.TAU_FLOOR, dist / 2)
        E = np.empty(n)
        choice = []
        cost = {"USD": 0.0, "HQC": 0.0}
        for k in range(n):
            pname, arm, si = self.choose_arm(tau[k])
            choice.append((pname, arm, self.cfg.shots[si]))
            E[k] = self.ev_disp[pname][arm][k, 0, si]
            cost[pricing.currency(pname)] += pricing.eval_cost(pname, arm, self.cfg.shots[si])
        dH = cl.E_hs[self.disp] - E
        r = self._policy_row(design.t_half(dH), dH)
        r.update(policy="routed, per-candidate tolerance", cost_usd=cost["USD"], cost_hqc=cost["HQC"])
        rows.append(r)
        from collections import Counter
        mix = Counter(choice)
        M["routedWrong"], M["routedHardWrong"], M["routedRmse"], M["routedTau"] = r["wrong"], r["hard_wrong"], r["rms"], r["tau"]
        M["routedUSD"], M["routedHQC"] = cost["USD"], cost["HQC"]
        M["routedMix"] = "; ".join(f"{v} to {C.PLATFORMS[p]['label']}, {a}, {s:,} shots" for (p, a, s), v in mix.most_common())
        self.R["tab_policies"] = rows
        self.R["route_tolerances"] = dict(min=float(tau.min()), max=float(tau.max()))

    def rsem_table(self):
        tab = []
        ne = self.cfg.rsem_epochs
        for pname in PLAT:
            for si, shots in enumerate(self.cfg.shots):
                row = dict(platform=pname, shots=shots)
                for arm in ("raw", "mit", "rsc", "pmit"):
                    e = self.all_errors(pname, arm)[:, :ne, si]
                    row[arm] = _rms(e)
                    row[arm + "_bias"] = float(e.mean())
                tab.append(row)
        self.R["tab_rsem"] = tab

    def caldec_table(self):
        """Surrogate-only decisions on the dispatched pairs after calibration with each arm, over noise draws."""
        sp, cl = self.sp, self.cl
        tab = []
        ne = self.cfg.rsem_epochs
        for pname in PLAT:
            for si, shots in enumerate(self.cfg.shots):
                row = dict(platform=pname, shots=shots)
                for arm in ("mit", "rsc", "pmit"):
                    wr = []
                    for e in range(ne):
                        fit = self.fit_surrogate(self.ev_cal[pname][arm][:, e, si])
                        ok = design.pair_success(sp, fit["T"])[self.dpairs]
                        wr.append(int(np.sum(ok != cl.ok_exact[self.dpairs])))
                    c = pricing.eval_cost(pname, arm, shots)
                    row[arm] = dict(mean=float(np.mean(wr)), max=int(np.max(wr)),
                                    cost=None if c is None else c * C.N_CAL)
                tab.append(row)
        self.R["tab_caldec"] = tab

    # ---------------------------------------------------------------- budget law
    def budget_law(self):
        t0 = time.time()
        cl = self.cl
        rng = self.rng("budget")
        ns = [5, 10, 20, 40, 75, 150]
        ne = self.cfg.rsem_epochs
        out = []
        for pname in PLAT:
            for arm in ("mit", "rsc", "pmit"):
                for si, shots in enumerate(self.cfg.shots):
                    E = self.ev_cal[pname][arm][:, :ne, si]
                    mse = []
                    for n in ns:
                        vals = []
                        for _ in range(self.cfg.budget_subsets):
                            sub = rng.choice(len(self.cal), n, replace=False)
                            ep = rng.integers(0, ne)
                            fit = self.fit_surrogate(E[sub, ep], cal=self.cal[sub])
                            e = fit["T"][self.heldout] - cl.T_exact[self.heldout]
                            vals.append(np.mean(e ** 2))
                        mse.append(float(np.mean(vals)))
                    # sigma^2(n) = b^2 + s^2/n, nonnegative least squares
                    from scipy.optimize import nnls
                    Amat = np.column_stack([np.ones(len(ns)), 1.0 / np.array(ns)])
                    (b2, s2), _ = nnls(Amat, np.array(mse))
                    out.append(dict(platform=pname, arm=arm, shots=shots, n=ns, mse=mse,
                                    floor=float(np.sqrt(b2)), per_point=float(np.sqrt(s2))))
        self.R["budget_law"] = out
        self.R["timing"]["budget_law"] = time.time() - t0

    # ---------------------------------------------------------------- reduced roughness ablation
    def hidden_basis(self):
        sp, cl = self.sp, self.cl
        types = list(C.ALL_LIGANDS.keys())
        tix = {t: i for i, t in enumerate(types)}
        pidx = {}
        for i in range(len(types)):
            for j in range(i, len(types)):
                pidx[(i, j)] = len(pidx)
        F = np.zeros((sp.n_complexes, len(pidx)))
        for c, sites in enumerate(sp.sites):
            ids = sorted(tix[s] for s in sites)
            for a in range(6):
                for b in range(a + 1, 6):
                    F[c, pidx[(ids[a], ids[b])]] += 1
        D, B, E2 = sp.Delta, sp.B, cl.E2
        Q = np.column_stack([np.ones_like(D), D, B, E2, D * D, B * B, E2 * E2, D * B, D * E2, B * E2])
        Q = Q / np.linalg.norm(Q, axis=0)
        return F, Q

    def hidden_term(self, seed, F, Q):
        w = np.random.default_rng([self.cfg.seed, 99, seed]).standard_normal(F.shape[1])
        h = F @ w
        hc = h - h.mean()
        coef, *_ = np.linalg.lstsq(Q, h, rcond=None)
        hp = h - Q @ coef
        absorbed = 1 - np.sum(hp ** 2) / np.sum(hc ** 2)
        return hp / np.sqrt(np.mean(hp ** 2)), float(absorbed)

    def roughness(self):
        t0 = time.time()
        cl, sp = self.cl, self.sp
        kpc = self.kpc()
        F, Q = self.hidden_basis()
        sigmas = np.array([0, 5, 10, 20, 30, 45, 70, 100, 150, 220, 330, 560.0])
        units = []
        absorbed = []
        for sd in range(self.cfg.rough_seeds):
            u, ab = self.hidden_term(sd, F, Q)
            units.append(u); absorbed.append(ab)
        ne = self.cfg.rsem_epochs
        law = {(r["platform"], r["arm"], r["shots"]): r for r in self.R["budget_law"]}
        rng = self.rng("rough")
        combos = []
        evalidx = np.concatenate([self.cal, self.disp])
        for pname in PLAT:
            for arm in ("mit", "rsc", "pmit"):
                for si, shots in enumerate(self.cfg.shots):
                    errE = np.concatenate([self.ev_cal[pname][arm], self.ev_disp[pname][arm]], 0)[:, :ne, si] \
                        - cl.E_ls[evalidx][:, None]                     # energy errors (cm^-1), (203, ne)
                    D = _rms(errE * kpc)
                    curve = np.zeros((len(units), len(sigmas)))
                    for ui, u in enumerate(units):
                        for li, sg in enumerate(sigmas):
                            h = u * sg / kpc            # cm^-1
                            truth = cl.E_ls + h
                            Ttruth = design.t_half(cl.E_hs - truth)
                            vals = []
                            for _ in range(self.cfg.rough_reps):
                                ep = rng.integers(0, ne, len(self.cal))
                                lab = truth[self.cal] + errE[np.arange(len(self.cal)), ep]
                                fit = self.fit_surrogate(lab, Ecorr_offset=None)
                                e = fit["T"][self.heldout] - Ttruth[self.heldout]
                                vals.append(np.mean(e ** 2))
                            curve[ui, li] = np.sqrt(np.mean(vals))
                    mean_curve = curve.mean(0)
                    above = np.where(mean_curve >= D)[0]
                    if len(above) and above[0] > 0:
                        j = above[0]
                        x0, x1, y0, y1 = sigmas[j - 1], sigmas[j], mean_curve[j - 1], mean_curve[j]
                        sstar = float(x0 + (D - y0) * (x1 - x0) / (y1 - y0))
                    elif len(above):
                        sstar = 0.0
                    else:
                        sstar = float("nan")
                    lw = law[(pname, arm, shots)]
                    pred2 = D ** 2 - lw["floor"] ** 2 - lw["per_point"] ** 2 / C.N_CAL
                    pred = float(np.sqrt(pred2)) if pred2 > 0 else 0.0
                    combos.append(dict(platform=pname, arm=arm, shots=shots, D=D, sigma_star=sstar,
                                       predicted=pred, curve=mean_curve.tolist()))
        ok = [c for c in combos if np.isfinite(c["sigma_star"]) and c["sigma_star"] > 0]
        rel = [abs(c["predicted"] - c["sigma_star"]) / c["sigma_star"] for c in ok]
        corr = float(np.corrcoef([c["predicted"] for c in ok], [c["sigma_star"] for c in ok])[0, 1]) if len(ok) > 2 else float("nan")
        M = self.M
        M["rgSigmaLevels"] = len(sigmas)
        M["rgSigmaMax"] = float(sigmas.max())
        M["rgSeeds"] = self.cfg.rough_seeds
        M["rgReps"] = self.cfg.rough_reps
        M["rgCombos"] = len(combos)
        M["rgCrossAll"] = len(ok)
        M["rgAbsorbed"] = 100 * float(max(absorbed))
        M["rgPredMed"] = 100 * float(np.median(rel)) if rel else float("nan")
        M["rgPredMax"] = 100 * float(np.max(rel)) if rel else float("nan")
        M["rgPredCorr"] = corr
        if ok:
            M["rgSigMin"] = float(min(c["sigma_star"] for c in ok))
            M["rgSigMax"] = float(max(c["sigma_star"] for c in ok))
            M["rgSigMed"] = float(np.median([c["sigma_star"] for c in ok]))
            fr = [c["sigma_star"] / c["D"] for c in ok]
            M["rgFracMed"], M["rgFracMin"], M["rgFracQlo"] = float(np.median(fr)), float(np.min(fr)), float(np.quantile(fr, 0.25))
        self.R["roughness"] = dict(sigmas=sigmas.tolist(), combos=combos, absorbed=absorbed)
        M["rgNcal"] = C.N_CAL
        for key, (pn, ar, sh) in (("rgEx", ("Helios", "rsc", 16000)), ("rgExCep", ("Cepheus", "rsc", 1000))):
            c = [x for x in combos if (x["platform"], x["arm"], x["shots"]) == (pn, ar, sh)]
            if c:
                M[key + "Dev"], M[key + "Sig"] = c[0]["D"], c[0]["sigma_star"]
                M[key + "Floor"] = law[(pn, ar, sh)]["floor"]
                if key == "rgEx":
                    M["rgExSmooth"] = c[0]["curve"][0]
        if ok:
            q = min(ok, key=lambda c: c["sigma_star"]); n_ = max(ok, key=lambda c: c["sigma_star"])
            M["rgQuietSig"], M["rgQuietDev"], M["rgQuietShots"] = q["sigma_star"], q["D"], q["shots"]
            M["rgNoisySig"], M["rgNoisyDev"], M["rgNoisyShots"] = n_["sigma_star"], n_["D"], n_["shots"]
            self.R["roughness"]["quiet"] = q["platform"] + "/" + q["arm"]
            self.R["roughness"]["noisy"] = n_["platform"] + "/" + n_["arm"]
        # decision-level comparison at 16k shots (one hidden-term draw), all 1204 pairs
        if 16000 in self.cfg.shots:
            si = self.cfg.shots.index(16000)
            dec = []
            u = units[0]
            for pname in PLAT:
                for arm in ("mit", "rsc", "pmit"):
                    errE = np.concatenate([self.ev_cal[pname][arm], self.ev_disp[pname][arm]], 0)[:, :ne, si] \
                        - cl.E_ls[evalidx][:, None]
                    pool = errE.ravel()
                    ws, wd = [], []
                    for sg in sigmas:
                        truth = cl.E_ls + u * sg / kpc
                        Ttruth = design.t_half(cl.E_hs - truth)
                        okt = design.pair_success(sp, Ttruth)
                        a_s, a_d = [], []
                        for _ in range(max(4, self.cfg.rough_reps // 4)):
                            ep = rng.integers(0, ne, len(self.cal))
                            lab = truth[self.cal] + errE[np.arange(len(self.cal)), ep]
                            fit = self.fit_surrogate(lab)
                            a_s.append(np.sum(design.pair_success(sp, fit["T"]) != okt))
                            dev = truth + rng.choice(pool, sp.n_complexes)
                            a_d.append(np.sum(design.pair_success(sp, design.t_half(cl.E_hs - dev)) != okt))
                        ws.append(float(np.mean(a_s))); wd.append(float(np.mean(a_d)))
                    ws, wd = np.array(ws), np.array(wd)
                    above = np.where(ws >= wd)[0]
                    cross = float("nan")
                    if len(above) and above[0] > 0:
                        j = above[0]
                        d0, d1 = ws[j - 1] - wd[j - 1], ws[j] - wd[j]
                        cross = float(sigmas[j - 1] + (0 - d0) * (sigmas[j] - sigmas[j - 1]) / (d1 - d0))
                    dec.append(dict(platform=pname, arm=arm, wrong_surr=ws.tolist(), wrong_dev=wd.tolist(),
                                    cross=cross))
            self.R["roughness"]["decisions_16k"] = dec
            M["rgWrongSurrZero"] = float(np.mean([d["wrong_surr"][0] for d in dec]))
            M["rgWrongDevZero"] = float(np.mean([d["wrong_dev"][0] for d in dec]))
            cr = [d["cross"] for d in dec if np.isfinite(d["cross"])]
            M["rgDecN"] = len(dec)
            if cr:
                M["rgDecMed"], M["rgDecMin"], M["rgDecMax"] = float(np.median(cr)), float(min(cr)), float(max(cr))
        self.R["timing"]["roughness"] = time.time() - t0

    # ---------------------------------------------------------------- eps sweep
    def eps_sweep(self):
        t0 = time.time()
        sp, cl = self.sp, self.cl
        shots_keep = self.cfg.shots
        rows = []
        for r2 in C.EPS_R2Q:
            p = dict(r2q=r2, r1q=C.EPS_R1Q, ro=C.EPS_RO, routed=False)
            z = Dv.drift_draws([self.cfg.seed, 11, int(r2 * 1e6)], len(self.disp), self.cfg.eps_epochs)
            rng = self.rng("eps", int(r2 * 1e6))
            ev = Dv.evaluate(sp.Delta[self.disp], sp.J[self.disp], self.theta_disp, p, z, C.EPS_SHOTS, rng,
                             twin=False, backend=self.cfg.backend)
            e = -(ev["mit"] - cl.E_ls[self.disp][:, None, None]) * self.kpc()
            row = dict(r2q=r2)
            for si, s in enumerate(C.EPS_SHOTS):
                row[f"rmse_{s}"] = _rms(e[..., si]); row[f"bias_{s}"] = float(e[..., si].mean())
            rows.append(row)
        self.R["tab_eps"] = rows
        names = {1e-4: "Ezerozeroonezero", 2e-4: "Ezerozerotwozero", 4e-4: "Ezerozerofourzero",
                 8e-4: "Ezerozeroeightzero", 16e-4: "Ezeroonesixzero", 32e-4: "Ezerothreetwozero"}
        for row in rows:
            nm = names[row["r2q"]]
            self.M[f"epsRmse{nm}Sonesixk"] = row["rmse_16000"]; self.M[f"epsBias{nm}Sonesixk"] = row["bias_16000"]
            self.M[f"epsRmse{nm}Ssixfourk"] = row["rmse_64000"]; self.M[f"epsBias{nm}Ssixfourk"] = row["bias_64000"]
        self.R["timing"]["eps"] = time.time() - t0

    # ---------------------------------------------------------------- sensitivity, scaling, extension
    def sensitivity(self):
        rows = []
        tagk = {0.20: "Ktwozero", 0.24: "Ktwofour", 0.28: "Ktwoeight"}
        tagd = {45: "dSfourfive", 60: "dSsixzero", 75: "dSsevenfive"}
        for k in (0.20, 0.24, 0.28):
            sp = L.Space(k=k)
            for dS in (45, 60, 75):
                cl = design.Classical(sp, dS=dS)
                okpt = design.pair_success(sp, cl.T_pt)
                row = dict(k=k, dS=dS, ok=int(cl.ok_exact.sum()), ls=100 * float(np.mean(cl.dH_exact > 0)),
                           agree_pt=100 * float(np.mean(okpt == cl.ok_exact)))
                rows.append(row)
                t = tagk[k] + tagd[dS]
                self.M[f"sensOk{t}"], self.M[f"sensLS{t}"], self.M[f"sensAgreePT{t}"] = row["ok"], row["ls"], row["agree_pt"]
        self.R["tab_sens"] = rows

    def scaling(self):
        spaces = [("five d orbitals", 5, 3), ("d + six donors", 11, 9), ("d + donors + pi*", 17, 9),
                  ("second d shell", 22, 9)]
        rows = []
        for name, n, p in spaces:
            ng = p * (n - p)
            n2 = 4 * ng
            fid = {pl: (1 - 4 * C.PLATFORMS[pl]["r2q"] / 3) ** n2 for pl in PLAT}
            rstar = 0.75 * (1 - 0.5 ** (1.0 / (2 * n2)))
            rows.append(dict(space=name, qubits=n, givens=ng, n2q=n2, fidelity=fid, r_star=rstar))
        self.R["tab_scaling"] = rows

    def extension(self):
        """Exact labels for the 4- and 5-type scaffolds (cyanide design), sec. 6.5 (labels only)."""
        ext = L.enumerate_extended_scaffolds()
        sp = L.Space(scaffolds=ext)
        cl = design.Classical(sp)
        cn = sp.pairs[:, 3] == C.SIXTH_SITE.index("CN")
        self.M["gnExt"] = len(ext)
        self.M["gnExtOk"] = int(cl.ok_exact[cn].sum())
        self.M["gnFracExt"] = 100 * float(cl.ok_exact[cn].mean())
        base_cn = self.sp.pairs[:, 3] == C.SIXTH_SITE.index("CN")
        self.M["gnBaseOk"] = int(self.cl.ok_exact[base_cn].sum())

    # ---------------------------------------------------------------- implemented stack (sec. 6.3-6.7)
    def stack_decoder(self):
        from . import stack
        t0 = time.time()
        model, info = stack.train_decoder(self, steps=self.cfg.rdm_steps)
        self.decoder = model
        rows = stack.decoder_table(self, model)
        summ = stack.summarise_decoder(rows, self.cfg.shots)
        self.R["tab_rdm"] = rows
        self.R["rdm_summary"] = summ
        self.R["rdm_train"] = info
        for r in rows:
            tag = C.SHOT_TAGS[self.cfg.shots.index(r["shots"])]
            self.M[f"rdRdm{r['platform']}{tag}"] = r["rdm"]
            self.M[f"rdPmit{r['platform']}{tag}"] = r["pmit"]
            self.M[f"rdMit{r['platform']}{tag}"] = r["mit"]
            self.M[f"rdDepth{r['platform']}{tag}"] = r["depth"]
        lo, hi = self.cfg.shots[0], self.cfg.shots[-1]
        self.M["rdSummary"] = (
            f"The decoder beats the paired-mitigated arm in {summ['beats_pmit']} of the {summ['n']} platform and shot "
            f"combinations, the extrapolated arm in {summ['beats_mit']} and the reference-scaled arm in {summ['beats_rsc']}. "
            f"At {lo} shots its error averaged over the five platforms is {summ[f'avg_{lo}']['rdm']:.0f} K against "
            f"{summ[f'avg_{lo}']['pmit']:.0f} K for paired mitigation, {summ[f'avg_{lo}']['mit']:.0f} K for extrapolation and "
            f"{summ[f'avg_{lo}']['rsc']:.0f} K for reference scaling; at {hi} shots the four figures are "
            f"{summ[f'avg_{hi}']['rdm']:.0f}, {summ[f'avg_{hi}']['pmit']:.0f}, {summ[f'avg_{hi}']['mit']:.0f} and "
            f"{summ[f'avg_{hi}']['rsc']:.0f} K.")
        self.R["timing"]["rdm_decoder"] = time.time() - t0

    def stack_surrogate(self):
        from . import stack
        t0 = time.time()
        out = stack.surrogate_study(self, self.decoder)
        self.R["rdm_surrogate"] = out
        M = self.M
        M["rdSurrDeviceLin"], M["rdSurrDeviceRdm"] = out["device_lin"], out["device_rdm"]
        M["rdSurrRdmLabLin"], M["rdSurrRdmLabRdm"] = out["rdmlab_lin"], out["rdmlab_rdm"]
        M["rdSurrExactLin"], M["rdSurrExactRdm"] = out["exact_lin"], out["exact_rdm"]
        M["rsSurrHeronPmitLo"] = out["pmit_lin"]
        M["rdSurrDepthMean"], M["rdSurrDepthCorr"] = out["depth_mean"], out["depth_corr_abs_ecorr"]
        self.R["timing"]["rdm_surrogate"] = time.time() - t0

    def stack_generative(self):
        from . import genrl
        g = genrl.generative(self.sp, self.cl)
        self.R["generative"] = g
        M = self.M
        M["gnHitPrior"] = 100 * g["hit_prior"]
        M["gnHitPriorUnique"] = 100 * g["hit_prior_unique"]
        M["gnNUnique"] = g["n_unique"]
        M["gnPrecTen"], M["gnPrecTwenty"], M["gnPrecForty"] = (100 * g["prec"][k] for k in (10, 20, 40))
        M["gnHitPriorScoredTwenty"] = 100 * g["prior_scored_top20"]
        M["gnBestName"] = g["best_name"]

    def stack_escalation(self):
        from . import genrl
        t0 = time.time()
        res = {p: genrl.escalation(self, p, episodes=self.cfg.rl_episodes) for p in ("Helios", "Heron")}
        self.R["escalation"] = res
        M = self.M
        h = res["Helios"]
        l1 = [x for x in h["learned"] if x["lam"] == 1.0][0]
        M["rlHeliosLearnMidCost"], M["rlHeliosLearnMidErr"] = l1["cost"], l1["err"]
        M["rlHeliosFixedHiCost"], M["rlHeliosFixedHiErr"] = h["fixed"][2]["cost"], h["fixed"][2]["err"]
        for pn, r in res.items():
            top = max(r["learned"], key=lambda x: x["lam"])
            fx = r["fixed"][-1]
            if pn == "Helios":
                M["rlHeliosFixedTopCost"], M["rlHeliosFixedTopErr"] = fx["cost"], fx["err"]
                M["rlHeliosLearnTopCost"], M["rlHeliosLearnTopErr"] = top["cost"], top["err"]
            M[f"rl{pn}Saving"] = 100 * (fx["cost"] - top["cost"]) / fx["cost"]
        self.R["timing"]["escalation"] = time.time() - t0

    def stack_qaoa(self, qasm_dir=None):
        from . import qaoa, pricing
        t0 = time.time()
        dist = np.minimum(np.abs(self.T2[self.disp] - C.T_LO), np.abs(self.T2[self.disp] - C.T_HI))
        pick = np.argsort(dist)[:qaoa.NQ]
        d = dist[pick]
        values = 1.0 - d / self.M["band"]
        costs = []
        for k in pick:
            pn, arm, si = self.choose_arm(max(C.TAU_FLOOR, dist[k] / 2))
            costs.append(pricing.eval_cost(pn, arm, self.cfg.shots[si]))
        costs = np.array(costs) / np.mean(costs)
        budget, penalty = qaoa.NQ / 3.0, 1.0
        f = qaoa.problem(values, costs, budget, penalty)
        res = qaoa.optimise(f, depths=(1, 2, 3), n_starts=8, seed=self.cfg.seed % 1000)
        const, h, J = qaoa.ising(values, costs, budget, penalty)
        files, ncnot = [], {}
        if qasm_dir:
            files, ncnot = qaoa.export(res, h, J, qasm_dir)
            dev = 0.0
            for p_, fn in zip(res["depths"], files):        # verify: re-simulate the exported circuits
                with open(os.path.join(qasm_dir, fn)) as fh:
                    pq, _ = qaoa.simulate_qasm(fh.read())
                d_ = res["depths"][p_]
                dev = max(dev, float(np.abs(pq - qaoa.probs(f, d_["gammas"], d_["betas"])).max()))
            res["qasm_max_prob_deviation"] = dev
            assert dev < 1e-9, dev
        # p=1 landscape with the batched kernel (figure)
        gg, bb = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, np.pi / 2, 41))
        land = qaoa.batch_expect(f, np.column_stack([gg.ravel(), bb.ravel()]), 1).reshape(gg.shape)
        res.update(candidates=[int(self.disp[k]) for k in pick], values=values.tolist(), costs=costs.tolist(),
                   budget=budget, penalty=penalty, qasm_files=files, cnots=ncnot,
                   landscape=dict(gamma=gg[0].tolist(), beta=bb[:, 0].tolist(), E=land.tolist()))
        self.R["qaoa"] = res
        M = self.M
        M["qaRatioOne"], M["qaRatioThree"] = res["depths"][1]["ratio"], res["depths"][3]["ratio"]
        M["qaProbOne"], M["qaProbThree"] = res["depths"][1]["p_opt"], res["depths"][3]["p_opt"]
        M["qaEvalsThree"] = res["depths"][3]["evals"]
        M["qaCnotThree"] = ncnot.get(3, 2 * int(np.count_nonzero(np.triu(J, 1))) * 3)
        M["qaBrute"] = res["brute"]
        self.R["timing"]["qaoa"] = time.time() - t0

    # ---------------------------------------------------------------- driver
    def run(self, out_dir=None):
        t0 = time.time()
        self._angle_cache = {}
        self.classical()
        self.router_predictions()
        self.calibrate()
        self.dispatched()
        self.noise_table()
        self.surr_table()
        self.policies()
        self.rsem_table()
        self.caldec_table()
        self.budget_law()
        self.roughness()
        self.eps_sweep()
        self.sensitivity()
        self.scaling()
        self.extension()
        self.stack_decoder()
        self.stack_surrogate()
        self.stack_generative()
        self.stack_escalation()
        self.stack_qaoa(qasm_dir=os.path.join(out_dir, "..", "qasm", "qaoa") if out_dir else None)
        self.R["disp"] = self.disp.tolist()
        self.R["cal"] = self.cal.tolist()
        self.R["T2"] = self.T2.tolist()
        self.R["T_exact"] = self.cl.T_exact.tolist()
        self.R["timing"]["total"] = time.time() - t0
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(self.R, f, indent=1, default=_json_default)
        return self.R


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
