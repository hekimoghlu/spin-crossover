"""Generative proposal beyond the enumerated space (sec. 6.5) and the Q-learned shot-escalation
policy (sec. 6.6)."""
from collections import Counter

import numpy as np

from . import constants as C
from . import design
from . import ligands as L
from . import pricing

SCAF = list(C.SCAFFOLD_LIGANDS.keys())


# ------------------------------------------------------------------ generative proposer
def _fractions(scaffolds):
    return np.array([[sc.count(l) / 5.0 for l in SCAF] for sc in scaffolds])


def _logistic_fit(X, y, l2=1e-2, steps=20000, lr=0.5):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return w


def _logistic_score(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def rust_extended_labels(threads=4):
    """Run the Rust candidate generator/scorer (std threads); None if it is not built."""
    import os, subprocess, tempfile
    exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rust", "target", "release",
                       "scospace")
    if not os.path.exists(exe) or os.environ.get("SCODESIGN_NO_NATIVE", "0") == "1":
        return None
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ext.csv")
        subprocess.run([exe, "--extended", out, str(threads)], check=True, capture_output=True)
        d = np.genfromtxt(out, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.asarray(d["ok_CN"], bool)


def generative(base_space, base_cl, n_samples=20000, seed=0):
    analyte = C.SIXTH_SITE.index("CN")
    # labels of the enumerated three-type space for the cyanide design
    sel = base_space.pairs[:, 3] == analyte
    ok_base = np.zeros(len(base_space.scaffolds), bool)
    ok_base[base_space.pairs[sel, 2]] = base_cl.ok_exact[sel]
    succ = [base_space.scaffolds[i] for i in np.where(ok_base)[0]]
    # unenumerated four- and five-type space, labelled exactly
    ext = L.enumerate_extended_scaffolds()
    ext_sp = L.Space(scaffolds=ext)
    ext_cl = design.Classical(ext_sp)
    s2 = ext_sp.pairs[:, 3] == analyte
    ok_ext = np.zeros(len(ext), bool)
    ok_ext[ext_sp.pairs[s2, 2]] = ext_cl.ok_exact[s2]
    rust_labels = rust_extended_labels()
    label_source = "python"
    rust_agree = None
    if rust_labels is not None:
        rust_agree = bool(len(rust_labels) == len(ok_ext) and np.array_equal(rust_labels, ok_ext))
        ok_ext = rust_labels                     # candidate generation + exact scoring done in Rust
        label_source = "rust"
    ext_key = {tuple(sc): i for i, sc in enumerate(ext)}
    # site-composition prior from ligand frequencies of the successful scaffolds
    cnt = Counter(l for sc in succ for l in sc)
    freq = np.array([cnt[l] for l in SCAF], float); freq /= freq.sum()
    rng = np.random.default_rng(seed)
    props = []
    order = {l: i for i, l in enumerate(SCAF)}
    while len(props) < n_samples:
        draw = rng.choice(len(SCAF), (4096, 5), p=freq)
        for d in draw:
            if len(set(d)) >= 4:
                props.append(ext_key[tuple(sorted((SCAF[k] for k in d), key=lambda l: order[l]))])
    props = np.array(props[:n_samples])
    uniq = np.unique(props)
    # logistic scorer on the seven ligand fractions, fitted to the enumerated space
    w = _logistic_fit(_fractions(base_space.scaffolds), ok_base.astype(float))
    score = _logistic_score(w, _fractions(ext))
    rank = np.argsort(-score)
    prec = {k: float(ok_ext[rank[:k]].mean()) for k in (10, 20, 40)}
    rank_prior = uniq[np.argsort(-score[uniq])]
    best = rank[0]
    name = "+".join(f"{ext[best].count(l)}{l}" for l in SCAF if ext[best].count(l))
    # hit-rate curve of the three strategies against number of distinct proposals (fig 6c)
    curve_n = np.arange(1, len(ext) + 1)
    uni_order = np.random.default_rng(seed + 1).permutation(len(ext))
    first_seen = []
    seen = set()
    for p in props:
        if p not in seen:
            seen.add(p); first_seen.append(p)
    first_seen = np.array(first_seen)
    return dict(
        label_source=label_source, rust_python_agree=rust_agree,
        n_ext=len(ext), n_ext_ok=int(ok_ext.sum()), frac_ext=float(ok_ext.mean()), n_base_ok=int(ok_base.sum()),
        hit_prior=float(ok_ext[props].mean()), n_unique=int(len(uniq)), hit_prior_unique=float(ok_ext[uniq].mean()),
        prec=prec, prior_scored_top20=float(ok_ext[rank_prior[:20]].mean()), best_name=name,
        best_ok=bool(ok_ext[best]), prior_freq=dict(zip(SCAF, freq.tolist())), weights=w.tolist(),
        curves=dict(n=curve_n.tolist(),
                    uniform=np.cumsum(ok_ext[uni_order]).tolist(),
                    prior=np.cumsum(ok_ext[first_seen]).tolist(),
                    scorer=np.cumsum(ok_ext[rank]).tolist()))


# ------------------------------------------------------------------ Q-learned escalation
BINS = np.array([0.5, 1.0, 2.0, 3.0])


def _state(level, dist, sig):
    return level, int(np.searchsorted(BINS, dist / sig[level]))


def escalation(loop, platform, lambdas=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0), arm="mit", episodes=40000, seed=0):
    """Tabular Q-learning: state = (shot level, distance to nearer threshold / that level's error);
    actions stop (decide three-way state vs 290/330 K) or escalate; reward = -cost of each evaluation
    (normalised by the full-escalation cost) - lambda per wrong state. Train epochs 0-3, test 4-7."""
    cl, kpc = loop.cl, loop.kpc()
    shots = loop.cfg.shots
    nL = len(shots)
    E = loop.ev_disp[platform][arm]                       # (n, ne, nL)
    T = design.t_half(cl.E_hs[loop.disp][:, None, None] - E)
    truth = design.three_state(cl.T_exact[loop.disp])
    dist = np.minimum(np.abs(T - C.T_LO), np.abs(T - C.T_HI))
    states = design.three_state(T)
    ne = E.shape[1]                                     # 8 epochs: train 0-3, test 4-7
    tr_ep, te_ep = list(range(ne // 2)), list(range(ne // 2, ne))
    err_T = T[:, tr_ep, :] - cl.T_exact[loop.disp][:, None, None]
    sig = np.sqrt(np.mean(err_T ** 2, axis=(0, 1)))       # per-level error estimated on training epochs
    cost = np.array([pricing.eval_cost(platform, arm, s) for s in shots])
    ctop = cost.sum()
    nb = len(BINS) + 1
    out = []

    def run_policy(Q, eps_list, greedy=True):
        tot_c, wrong = [], []
        for i in range(len(loop.disp)):
            for e in eps_list:
                lv = 0; c = cost[0]
                while True:
                    s = _state(lv, dist[i, e, lv], sig)
                    a = 0 if lv == nL - 1 else int(np.argmax(Q[s]))
                    if a == 0:
                        break
                    lv += 1; c += cost[lv]
                tot_c.append(c); wrong.append(states[i, e, lv] != truth[i])
        return float(np.mean(tot_c)), float(np.mean(wrong))

    rng = np.random.default_rng([loop.cfg.seed, 71, seed])
    for lam in lambdas:
        Q = np.zeros((nL, nb, 2))
        Q[nL - 1, :, 1] = -1e9
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.6 * episodes))
            alpha = 0.1
            i = rng.integers(len(loop.disp)); e = tr_ep[rng.integers(len(tr_ep))]
            lv = 0
            while True:
                s = _state(lv, dist[i, e, lv], sig)
                if lv == nL - 1:
                    a = 0
                else:
                    a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
                if a == 0:
                    r = -lam * float(states[i, e, lv] != truth[i])
                    Q[s][0] += alpha * (r - Q[s][0])
                    break
                r = -cost[lv + 1] / ctop
                s2 = _state(lv + 1, dist[i, e, lv + 1], sig)
                Q[s][1] += alpha * (r + np.max(Q[s2]) - Q[s][1])
                lv += 1
        c, w = run_policy(Q, te_ep)
        out.append(dict(lam=lam, cost=c, err=w))
    fixed = []
    for L_ in range(nL):
        c = float(cost[:L_ + 1].sum())
        w = float(np.mean(states[:, te_ep, L_] != truth[:, None]))
        fixed.append(dict(shots=shots[L_], cost=c, err=w))
    return dict(platform=platform, learned=out, fixed=fixed, sigma_levels=sig.tolist(), cost_levels=cost.tolist())
