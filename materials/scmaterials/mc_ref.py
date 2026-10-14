"""Pure-Python reference of the Rust Monte Carlo (rust/src/lib.rs, module mc).

Same lattice construction, site order, SplitMix64 stream and Metropolis rule,
so that small systems can be compared trajectory-for-trajectory.  Also a thin
wrapper that runs the Rust binary.
"""
from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path

R_GAS = 8.314462618
_M64 = (1 << 64) - 1
RUST_BIN = Path(__file__).resolve().parent.parent / "rust" / "target" / "release" / "scmc"


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & _M64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        return z ^ (z >> 31)

    def next_f64(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


def build_particle(d_nm, a_nm=0.73, delta_nm=1.0):
    r_act = d_nm / 2.0 - delta_nm
    m = math.ceil(d_nm / 2.0 / a_nm) + 1
    index = {}
    coords = []
    if r_act > 0:
        for i in range(-m, m + 1):
            for j in range(-m, m + 1):
                for k in range(-m, m + 1):
                    r = a_nm * math.sqrt(float(i * i + j * j + k * k))
                    if r <= r_act:
                        index[(i, j, k)] = len(coords)
                        coords.append((i, j, k))
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    nbr = [[index.get((i + a, j + b, k + c), -1) for a, b, c in dirs] for i, j, k in coords]
    return nbr


def hysteresis(nbr, gamma_bulk, t_half=310.0, ds=60.0, t_min=250.0, t_max=370.0,
               n_t=61, sweeps=100, seed=1):
    jc = gamma_bulk / 12.0
    dh = t_half * ds
    rng = SplitMix64(seed)
    ts = [t_min + (t_max - t_min) * k / (n_t - 1) for k in range(n_t)]
    s = [-1] * len(nbr)

    def sweep(t, h):
        rt = R_GAS * t
        for i in range(len(nbr)):
            tot = 0.0
            for j in nbr[i]:
                if j >= 0:
                    tot += float(s[j])
            si = float(s[i])
            de = 2.0 * jc * si * tot - 2.0 * h * si
            u = rng.next_f64()
            if de <= 0.0 or u < math.exp(-de / rt):
                s[i] = -s[i]

    frac = lambda: (sum(1 for v in s if v > 0) / len(s)) if s else float("nan")
    up = []
    for t in ts:
        h = 0.5 * (dh - t * ds)
        for _ in range(sweeps):
            sweep(t, h)
        up.append(frac())
    down = [0.0] * n_t
    for k in range(n_t - 1, -1, -1):
        t = ts[k]
        h = 0.5 * (dh - t * ds)
        for _ in range(sweeps):
            sweep(t, h)
        down[k] = frac()
    return ts, up, down


def run_rust_loop(out_csv, d, gamma, a=0.73, delta=1.0, tmin=250.0, tmax=370.0, nt=61, sweeps=100, seed=1):
    subprocess.run([str(RUST_BIN), "mcloop", str(out_csv), str(d), str(gamma), str(a), str(delta),
                    str(tmin), str(tmax), str(nt), str(sweeps), str(seed)], check=True)
    rows = list(csv.reader(open(out_csv)))
    n = int(rows[0][1])
    data = [[float(v) for v in r] for r in rows[2:]]
    return n, [r[0] for r in data], [r[1] for r in data], [r[2] for r in data]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))
