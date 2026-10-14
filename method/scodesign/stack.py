"""Recurrent-depth decoder (sec. 6.3) and recurrent-depth surrogate (sec. 6.4), run on the loop's data."""
import numpy as np

from . import ansatz as A
from . import constants as C
from . import design
from .rdm import RecurrentDepth, decoder_features

PLAT = list(C.PLATFORMS.keys())


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def _stack(loop, pname, key):
    return np.concatenate([loop.ev_cal[pname][key], loop.ev_disp[pname][key]], 0)


def train_decoder(loop, n_partners=8, epochs=None, steps=3000, seed=0):
    """Train on twin circuits only: twin A is the 'circuit', twin B (same platform, epoch, shots) the reference."""
    cl = loop.cl
    epochs = list(range(loop.cfg.rsem_epochs)) if epochs is None else epochs
    idx = np.concatenate([loop.cal, loop.disp])
    Eref = cl.E_ref[idx]
    rng = np.random.default_rng([loop.cfg.seed, 61])
    Xs, ys = [], []
    for pname in PLAT:
        ncnot = A.gate_counts(C.PLATFORMS[pname]["routed"], 1.0)[0]
        RS = _stack(loop, pname, "ref_scales")            # (n, ne, nshots, 3)
        for si, shots in enumerate(loop.cfg.shots):
            for e in epochs:
                Es = RS[:, e, si, :]
                n = len(Es)
                a = np.repeat(np.arange(n), n_partners)
                b = rng.integers(0, n - 1, len(a)); b = b + (b >= a)
                Xs.append(decoder_features(Es[a], Es[b], Eref[b], ncnot, shots))
                ys.append(Es[a, 0] / Eref[a])              # attenuation of twin A (known exactly)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    model = RecurrentDepth(X.shape[1], n_hidden=16, seed=seed)
    hist = model.fit(X, y, T_train=10, steps=steps, lr=3e-3, batch=1024, seed=seed)
    return model, dict(n_train=int(len(X)), loss_history=hist)


def decode(loop, model, pname, which, si, epoch):
    """Decoder energies for complexes `which` ('cal' or 'disp') on one platform/shot/epoch."""
    ev = loop.ev_cal if which == "cal" else loop.ev_disp
    idx = loop.cal if which == "cal" else loop.disp
    ncnot = A.gate_counts(C.PLATFORMS[pname]["routed"], 1.0)[0]
    Ea = ev[pname]["scales"][:, epoch, si, :]
    Er = ev[pname]["ref_scales"][:, epoch, si, :]
    X = decoder_features(Ea, Er, loop.cl.E_ref[idx], ncnot, loop.cfg.shots[si])
    att, depth = model.predict(X, tol=1e-3, T_max=64)
    return Ea[:, 0] / att, depth


def decoder_table(loop, model):
    cl, kpc = loop.cl, loop.kpc()
    ne = loop.cfg.rsem_epochs
    rows = []
    for pname in PLAT:
        for si, shots in enumerate(loop.cfg.shots):
            errs = {k: [] for k in ("mit", "rsc", "pmit", "rdm")}
            depths = []
            for e in range(ne):
                E, d = decode(loop, model, pname, "disp", si, e)
                errs["rdm"].append(-(E - cl.E_ls[loop.disp]) * kpc)
                depths.append(d)
                for arm in ("mit", "rsc", "pmit"):
                    errs[arm].append(-(loop.ev_disp[pname][arm][:, e, si] - cl.E_ls[loop.disp]) * kpc)
            row = dict(platform=pname, shots=shots, depth=float(np.mean(depths)))
            for k, v in errs.items():
                row[k] = _rms(np.concatenate(v))
            rows.append(row)
    return rows


def summarise_decoder(rows, shots):
    s = {}
    s["beats_pmit"] = sum(r["rdm"] < r["pmit"] for r in rows)
    s["beats_mit"] = sum(r["rdm"] < r["mit"] for r in rows)
    s["beats_rsc"] = sum(r["rdm"] < r["rsc"] for r in rows)
    s["n"] = len(rows)
    for sh in (shots[0], shots[-1]):
        rr = [r for r in rows if r["shots"] == sh]
        s[f"avg_{sh}"] = {k: float(np.mean([r[k] for r in rr])) for k in ("rdm", "pmit", "mit", "rsc")}
    s["depth_by_platform"] = {p: float(np.mean([r["depth"] for r in rows if r["platform"] == p])) for p in PLAT}
    return s


# ------------------------------------------------------------------ recurrent-depth surrogate
def surrogate_features(loop):
    sp, cl = loop.sp, loop.cl
    types = list(C.ALL_LIGANDS.keys())
    frac = np.zeros((sp.n_complexes, len(types)))
    for i, sites in enumerate(sp.sites):
        for s in sites:
            frac[i, types.index(s)] += 1 / 6.0
    return np.column_stack([frac, sp.Delta, sp.B, cl.E2])


def rdm_surrogate(loop, labels_cal, n_models=5, steps=3000):
    """Ensemble of recurrent-depth surrogates for E_corr from the calibration labels."""
    X = surrogate_features(loop)
    preds, depths = [], []
    for k in range(n_models):
        m = RecurrentDepth(X.shape[1], n_hidden=16, seed=100 + k)
        m.fit(X[loop.cal], labels_cal, T_train=10, steps=steps, lr=3e-3, batch=10 ** 9, l2=1e-4, seed=k,
              val_frac=0.2)
        p, d = m.predict(X, tol=1e-3, T_max=64)
        preds.append(p); depths.append(d)
    return np.mean(preds, 0), np.mean(depths, 0)


def heldout_rms_from_ecorr(loop, Ecorr_pred):
    cl = loop.cl
    T = design.t_half(cl.E_hs - (cl.E_ref + Ecorr_pred))
    return _rms(T[loop.heldout] - cl.T_exact[loop.heldout])


def surrogate_study(loop, model, epoch=1):
    """Line vs recurrent-depth ensemble on device-extrapolated, decoder and exact labels (Heron mit 1k by
    default: the deployed calibration arm, independent drift epoch)."""
    cl = loop.cl
    pname, arm, si = loop.cal_arm
    out = {}
    lab_dev = loop.ev_cal[pname][arm][:, epoch, si] - cl.E_ref[loop.cal]
    E_dec, _ = decode(loop, model, pname, "cal", si, epoch)
    lab_dec = E_dec - cl.E_ref[loop.cal]
    lab_pmit = loop.ev_cal[pname]["pmit"][:, epoch, si] - cl.E_ref[loop.cal]
    lab_exact = cl.Ecorr[loop.cal]
    for name, lab in (("device", lab_dev), ("rdmlab", lab_dec), ("exact", lab_exact), ("pmit", lab_pmit)):
        fit = loop.fit_surrogate(lab + cl.E_ref[loop.cal])
        out[name + "_lin"] = _rms(fit["T"][loop.heldout] - cl.T_exact[loop.heldout])
        if name != "pmit":
            pred, depth = rdm_surrogate(loop, lab)
            out[name + "_rdm"] = heldout_rms_from_ecorr(loop, pred)
            if name == "device":
                out["depth_mean"] = float(depth.mean())
                out["depth_corr_abs_ecorr"] = float(np.corrcoef(depth, np.abs(cl.Ecorr))[0, 1])
    out["arm"] = dict(platform=pname, arm=arm, shots=loop.cfg.shots[si], epoch=epoch)
    return out
