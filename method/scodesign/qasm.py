"""OpenQASM 3 export of the dispatched ansatz circuits (native and linear-chain routed), their
reference-state twins, the three measurement bases, and a Hamiltonian/Pauli manifest.

Verification re-simulates every exported file with an independent mini-parser (noiseless
statevector) and reconstructs the energy from the manifest coefficients.
(The manuscript ships OpenQASM 2.0; this reimplementation writes OpenQASM 3.0 as requested.)
"""
import json
import os
import re

import numpy as np

from . import ansatz as A
from . import kanamori as K
from . import ligands as L
from . import constants as C

BASES = ("Z", "X", "Y")


def circuit_qasm(thetas, routed, basis, title):
    ops = A.logical_gate_list(routed)
    wire = (lambda q: A.LAYOUT[q]) if routed else (lambda q: q)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"// {title}",
             "// qubit/orbital convention: logical qubit m = pair orbital m (0,1,2 = t2g; 3,4 = eg);",
             "// |1> = doubly occupied; classical bit c[m] holds logical qubit m."]
    if routed:
        lines.append("// routed on linear chain 0-3-1-4-2: logical m lives on physical q[%s]"
                     % ",".join(str(A.LAYOUT[m]) for m in range(5)))
    lines += ["qubit[5] q;", "bit[5] c;", "reset q;"]
    for m in (0, 1, 2):
        lines.append(f"x q[{wire(m)}];")
    for op in ops:
        if op[0] == A.OP_CNOT:
            lines.append(f"cx q[{op[1]}], q[{op[2]}];")
        elif op[0] == A.OP_RY:
            lines.append(f"ry({op[4] * thetas[op[3]]:.17g}) q[{op[1]}];")
    for m in range(5):
        w = wire(m)
        if basis == "X":
            lines.append(f"h q[{w}];")
        elif basis == "Y":
            lines.append(f"sdg q[{w}];")
            lines.append(f"h q[{w}];")
    for m in range(5):
        lines.append(f"c[{m}] = measure q[{wire(m)}];")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ independent mini-parser
_G1 = {
    "x": np.array([[0, 1], [1, 0]], complex),
    "h": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "sdg": np.diag([1, -1j]),
}


def parse_and_simulate(text):
    """Return (probabilities over the classical register c[4..0] as index with c[0] MSB, counts)."""
    psi = np.zeros(32, complex); psi[0] = 1
    meas = {}
    n2 = n1 = 0
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "qubit", "bit", "reset")):
            continue
        m = re.fullmatch(r"cx q\[(\d)\], q\[(\d)\];", line)
        if m:
            psi = A.apply_cnot(psi, int(m.group(1)), int(m.group(2))); n2 += 1; continue
        m = re.fullmatch(r"ry\(([-+0-9.eE]+)\) q\[(\d)\];", line)
        if m:
            psi = A.apply_1q(psi, A.ry(float(m.group(1))).astype(complex), int(m.group(2))); n1 += 1; continue
        m = re.fullmatch(r"(x|h|sdg) q\[(\d)\];", line)
        if m:
            psi = A.apply_1q(psi, _G1[m.group(1)], int(m.group(2))); continue
        m = re.fullmatch(r"c\[(\d)\] = measure q\[(\d)\];", line)
        if m:
            meas[int(m.group(1))] = int(m.group(2)); continue
        raise ValueError("unparsed line: " + raw)
    p_wire = np.abs(psi) ** 2
    p = np.zeros(32)
    for xw in range(32):
        xc = 0
        for cb, w in meas.items():
            if (xw >> (4 - w)) & 1:
                xc |= 1 << (4 - cb)
        p[xc] += p_wire[xw]
    return p, n2, n1


def energy_from_manifest_probs(probs, entry):
    E = entry["c0"]
    for s, c in entry["terms"]:
        kind = [ch for ch in s if ch != "I"][0]
        p = probs[kind]
        idx = [i for i, ch in enumerate(s) if ch != "I"]
        val = sum(p[x] * (-1) ** sum((x >> (4 - i)) & 1 for i in idx) for x in range(32))
        E += c * val
    return E


def export_and_verify(results_path, out_dir):
    with open(results_path) as f:
        R = json.load(f)
    sp = L.Space()
    disp = R["disp"]
    os.makedirs(out_dir, exist_ok=True)
    manifest = dict(convention="qubit m = pair orbital m; |1> = doubly occupied; bit c[m] = qubit m; "
                               "energy = c0 + sum coeff * <Pauli>; Z/ZZ from basis Z, XX from X, YY from Y; cm^-1",
                    complexes=[])
    nfiles = 0
    max_dev = 0.0
    counts = {}
    for k, i in enumerate(disp):
        D, J = sp.Delta[i], sp.J[i]
        th, en = A.optimise_angles(K.hamiltonian_matrix(D, J), n_starts=3, seed=int(i))
        c0, terms = K.pauli_terms(D, J)
        entry = dict(index=int(i), name=sp.name(i), Delta=float(D), B=float(sp.B[i]), J_H=float(J),
                     thetas=th.tolist(), c0=c0, terms=terms, E_ref=float(K.e_ref(D, J)),
                     E_hs=float(K.e_hs(D, J)), E_ls_exact=float(K.exact_ls(D, J)[0]), E_ansatz=float(en),
                     files={})
        for routed in (False, True):
            for circ, angles in (("ansatz", th), ("twin", np.zeros(6))):
                probs = {}
                for b in BASES:
                    tag = f"c{int(i):04d}_{circ}_{'routed' if routed else 'native'}_{b}.qasm"
                    txt = circuit_qasm(angles, routed, b, f"{entry['name']} {circ} basis {b}")
                    with open(os.path.join(out_dir, tag), "w") as f:
                        f.write(txt)
                    nfiles += 1
                    entry["files"].setdefault(f"{circ}_{'routed' if routed else 'native'}", []).append(tag)
                    p, n2, n1 = parse_and_simulate(txt)
                    if routed:   # structural check: every CNOT acts on chain neighbours
                        for a_, b_ in re.findall(r"cx q\[(\d)\], q\[(\d)\];", txt):
                            assert abs(int(a_) - int(b_)) == 1, (tag, a_, b_)
                    probs[b] = p
                    counts[(routed, circ)] = (n2, n1)
                E = energy_from_manifest_probs(probs, entry)
                ref = en if circ == "ansatz" else entry["E_ref"]
                max_dev = max(max_dev, abs(E - ref))
        manifest["complexes"].append(entry)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    report = dict(routed_nearest_neighbour_only=True, n_files=nfiles, n_native_files=nfiles // 2, max_energy_deviation_cm=max_dev,
                  gate_counts={f"{'routed' if r else 'native'}_{c}": v for (r, c), v in counts.items()})
    with open(os.path.join(out_dir, "verification.json"), "w") as f:
        json.dump(report, f, indent=1)
    assert max_dev < 1e-6, max_dev
    assert counts[(False, "ansatz")][0] == 24 and counts[(True, "ansatz")][0] == 48
    print(f"[qasm] wrote {nfiles} circuits; re-simulated energy deviation {max_dev:.2e} cm^-1")
    return report
