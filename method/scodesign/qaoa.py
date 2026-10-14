"""Twelve-qubit QAOA dispatch test (manuscript sec. 6.7).

Knapsack-style choice of which of the twelve candidates nearest the thresholds to evaluate:
    f(x) = sum_i v_i x_i - P (sum_i c_i x_i - B)^2,   x_i in {0,1}   (maximised)
v_i = closeness to the nearer threshold, c_i = evaluation cost (normalised), B = soft budget
(ASSUMPTIONS.md section 32). Cost table: NASM weighted_bitsum. Statevector: C++ kernel (qaoa_expect,
qaoa_batch) with a NumPy fallback. Parameters: Nelder-Mead from eight random starts, plus a warm
start from the previous depth for p > 1.
"""
import os
import re

import numpy as np
from scipy.optimize import minimize

from . import native

NQ = 12


def problem(values, costs, budget, penalty):
    n = len(values)
    vs = native.weighted_bitsum(n, values)
    cs = native.weighted_bitsum(n, costs)
    return vs - penalty * (cs - budget) ** 2


def ising(values, costs, budget, penalty):
    """f(x) = const + sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j with x_i = (1 - Z_i)/2."""
    v, c = np.asarray(values, float), np.asarray(costs, float)
    n = len(v)
    # linear part in x: a_i = v_i - P (c_i^2 - 2 B c_i) ; quadratic: -2 P c_i c_j for i<j
    a = v - penalty * (c ** 2 - 2 * budget * c)
    Q = -2 * penalty * np.outer(c, c)
    const = -penalty * budget ** 2 + a.sum() / 2 + np.sum(np.triu(Q, 1)) / 4
    h = -a / 2 - np.array([np.sum(np.triu(Q, 1)[i, :]) + np.sum(np.triu(Q, 1)[:, i]) for i in range(n)]) / 4
    J = np.triu(Q, 1) / 4
    return const, h, J


# ------------------------------------------------------------------ simulation
def _numpy_state(f, gammas, betas):
    N = len(f); n = int(np.log2(N))
    psi = np.full(N, 1 / np.sqrt(N), complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * f)
        t = psi.reshape([2] * n)
        c, s = np.cos(b), np.sin(b)
        for q in range(n):
            t0 = np.take(t, 0, axis=q); t1 = np.take(t, 1, axis=q)
            t = np.stack([c * t0 - 1j * s * t1, -1j * s * t0 + c * t1], axis=q)
        psi = t.reshape(-1)
    return psi


def expect(f, gammas, betas, backend="auto"):
    if backend in ("auto", "cpp"):
        r = native.qaoa_expect_native(NQ if len(f) == 4096 else int(np.log2(len(f))), f, gammas, betas)
        if r is not None:
            return r
    p = np.abs(_numpy_state(f, gammas, betas)) ** 2
    return float(p @ f)


def probs(f, gammas, betas, backend="auto"):
    n = int(np.log2(len(f)))
    if backend in ("auto", "cpp"):
        r = native.qaoa_expect_native(n, f, gammas, betas, want_probs=True)
        if r is not None:
            return r[1]
    return np.abs(_numpy_state(f, gammas, betas)) ** 2


def batch_expect(f, params, p, backend="auto"):
    n = int(np.log2(len(f)))
    if backend in ("auto", "cpp"):
        r = native.qaoa_batch_native(n, f, params, p)
        if r is not None:
            return r
    if backend == "cuda":
        from . import cuda_backend
        return cuda_backend.qaoa_batch(f, params, p)
    return np.array([expect(f, x[:p], x[p:], backend="numpy") for x in params])


# ------------------------------------------------------------------ optimisation
def optimise(f, depths=(1, 2, 3), n_starts=8, seed=0, backend="auto"):
    rng = np.random.default_rng(seed)
    fmax, fmin = f.max(), f.min()
    opt = np.where(f >= fmax - 1e-9)[0]
    res = {}
    prev = None
    for p in depths:
        evals = [0]

        def obj(x):
            evals[0] += 1
            return -expect(f, x[:p], x[p:], backend)
        starts = [np.r_[rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)] for _ in range(n_starts)]
        if prev is not None:
            g, b = prev[:p - 1], prev[p - 1:]
            starts.append(np.r_[g, g[-1], b, b[-1]])
        best = None
        for x0 in starts:
            r = minimize(obj, x0, method="Nelder-Mead", options=dict(maxiter=400 * 2 * p, xatol=1e-4, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
        x = best.x
        pr = probs(f, x[:p], x[p:], backend)
        e = float(pr @ f)
        res[p] = dict(gammas=x[:p].tolist(), betas=x[p:].tolist(), expectation=e,
                      ratio=e / fmax, ratio_minmax=(e - fmin) / (fmax - fmin), p_opt=float(pr[opt].sum()), evals=evals[0])
        prev = x
    return dict(depths=res, fmax=float(fmax), fmin=float(fmin), n_optimal=int(len(opt)),
                optimal_sets=[int(z) for z in opt], brute=int(len(f)))


# ------------------------------------------------------------------ OpenQASM 3 export and verification
def circuit_qasm(h, J, gammas, betas, title):
    n = len(h)
    L = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"// {title}",
         "// QAOA for the dispatch knapsack; cost f = const + sum h Z + sum J ZZ, phase exp(-i gamma f)",
         f"qubit[{n}] q;", f"bit[{n}] c;", "reset q;"]
    L += [f"h q[{i}];" for i in range(n)]
    for g, b in zip(gammas, betas):
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 0:
                    L += [f"cx q[{i}], q[{j}];", f"rz({2 * g * J[i, j]:.17g}) q[{j}];", f"cx q[{i}], q[{j}];"]
        for i in range(n):
            L.append(f"rz({2 * g * h[i]:.17g}) q[{i}];")
        for i in range(n):
            L.append(f"rx({2 * b:.17g}) q[{i}];")
    L += [f"c[{i}] = measure q[{i}];" for i in range(n)]
    return "\n".join(L) + "\n"


def simulate_qasm(text):
    """Independent parser for the QAOA subset (h, cx, rz, rx, measure); returns probabilities, CNOT count."""
    n = int(re.search(r"qubit\[(\d+)\]", text).group(1))
    psi = np.zeros(1 << n, complex); psi[0] = 1
    t = psi.reshape([2] * n)
    ncx = 0
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def one(t, U, q):
        return np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "qubit", "bit", "reset")):
            continue
        m = re.fullmatch(r"h q\[(\d+)\];", line)
        if m:
            t = one(t, H, int(m.group(1))); continue
        m = re.fullmatch(r"(rz|rx)\(([-+0-9.eE]+)\) q\[(\d+)\];", line)
        if m:
            th = float(m.group(2))
            U = (np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)]) if m.group(1) == "rz" else
                 np.array([[np.cos(th / 2), -1j * np.sin(th / 2)], [-1j * np.sin(th / 2), np.cos(th / 2)]]))
            t = one(t, U, int(m.group(3))); continue
        m = re.fullmatch(r"cx q\[(\d+)\], q\[(\d+)\];", line)
        if m:
            a, b = int(m.group(1)), int(m.group(2)); ncx += 1
            idx = [slice(None)] * n; idx[a] = 1
            sub = t[tuple(idx)]
            bb = b - (1 if b > a else 0)
            t = t.copy(); t[tuple(idx)] = np.flip(sub, axis=bb); continue
        if re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line):
            continue
        raise ValueError(raw)
    return np.abs(t.reshape(-1)) ** 2, ncx


def export(res, h, J, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files, ncnot = [], {}
    for p, d in res["depths"].items():
        name = f"qaoa_p{p}.qasm"
        txt = circuit_qasm(h, J, d["gammas"], d["betas"], f"QAOA depth {p}")
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(txt)
        files.append(name)
        ncnot[p] = txt.count("cx q[")
    return files, ncnot
