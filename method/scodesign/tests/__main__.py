import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

from .. import ansatz as A
from .. import constants as C
from .. import design
from .. import kanamori as K
from .. import ligands as L
from .. import native
from .. import noise as N
from .. import pricing
from .. import qasm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQUIRE_NATIVE = os.environ.get("SCODESIGN_REQUIRE_NATIVE", "1") == "1"


def test_space_counts():
    sp = L.Space()
    assert len(sp.scaffolds) == 301 and sp.n_complexes == 1505 and sp.n_pairs == 1204
    D, B = L.ligand_field(C.CAL_COMPOUND)
    assert abs(D - 10000 * (4 * 1.33 + 2 * 1.02) / 6) < 1e-9
    assert abs(L.pairing_energy(1.0) - 19.3) < 1e-12


def test_pauli_decomposition():
    D, J = 11500.0, 3300.0
    c0, terms = K.pauli_terms(D, J)
    assert len(terms) == 35 and abs(c0) < 1e-9
    g = K.measurement_groups(terms)
    assert sorted(len(v) for v in g.values()) == [10, 10, 15] and len(g) == 3
    H = K.hamiltonian_matrix(D, J)
    blk = H[np.ix_(K.BLOCK_STATES, K.BLOCK_STATES)].real
    assert np.abs(blk - K.hb_block(D, J)[0]).max() < 1e-9
    # block is invariant: no coupling out of the three-pair sector
    other = np.setdiff1d(np.arange(32), K.BLOCK_STATES)
    assert np.abs(H[np.ix_(other, K.BLOCK_STATES)]).max() < 1e-12


def test_full_space_vs_block():
    for D, J in ((10500.0, 3500.0), (13000.0, 3300.0)):
        chk = K.full_space_check(D, J)
        assert abs(chk["e_singlet"] - K.exact_ls(D, J)[0]) < 1e-8
        assert abs(chk["e_quintet"] - K.e_hs(D, J)) < 1e-8


def test_mf_pt2_closed_form():
    D, J = np.array([11000.0, 12500.0]), np.array([3200.0, 3400.0])
    assert np.allclose(K.dH_mf(D, J), 2 * D - 8 * J)
    assert np.allclose(K.e_pt2(D, J), -3 * J ** 2 / D)


def test_ansatz_compilation():
    th = np.random.default_rng(3).normal(0, 0.4, 6)
    for routed in (False, True):
        psi = A.run_statevector(A.logical_gate_list(routed), th, A.initial_index(routed))
        perm = A.wire_to_logical_perm(routed)
        pl = np.zeros(32, complex); pl[perm] = psi
        assert abs(abs(np.vdot(pl, A.ansatz_state(th))) - 1) < 1e-12
    assert A.gate_counts(False) == (24, 12) and A.gate_counts(True) == (48, 12)
    assert A.gate_counts(False, 2.0)[0] == 48 and A.gate_counts(True, 1.5)[0] == 72


def test_cpp_matches_numpy():
    if native.dm_lib() is None:
        assert not REQUIRE_NATIVE, "C++ kernel not built (make -C cpp)"
        print("   (C++ kernel not built; skipped)"); return
    rng = np.random.default_rng(7)
    for routed in (False, True):
        for s in C.ZNE_SCALES:
            code, coef = A.noisy_program(routed, s)
            params = np.column_stack([rng.normal(0, .3, (40, 6)), rng.uniform(0, .02, 40), rng.uniform(0, .05, 40)])
            a = N.run_batch(code, coef, params, A.initial_index(routed), backend="cpp")
            b = N.run_batch_numpy(code, coef, params, A.initial_index(routed))
            assert np.abs(a - b).max() < 1e-9, np.abs(a - b).max()
            assert np.allclose(np.trace(a, axis1=1, axis2=2), 1.0)


def test_nasm_parity():
    if native.par_lib() is None:
        assert not REQUIRE_NATIVE, "NASM kernel not built (make -C asm)"
        print("   (NASM kernel not built; skipped)"); return
    p = np.random.default_rng(1).random((13, 32))
    m = N.Z_MASKS + N.PAIR_MASKS
    assert np.abs(native.parity_expect(p, m) - native.parity_expect(p, m, force_numpy=True)).max() < 1e-12
    assert list(native.enum_popcount(5, 3)) == list(K.BLOCK_STATES)
    assert len(native.enum_popcount(10, 6)) == 210


def test_noiseless_energy_and_readout():
    D, J = 12266.7, 3226.0
    th, e = A.optimise_angles(K.hamiltonian_matrix(D, J))
    co = K.pauli_coeff_arrays(D, J)
    for routed in (False, True):
        code, coef = A.noisy_program(routed, 1.0)
        rho = N.run_batch(code, coef, np.array([list(th) + [0, 0]]), A.initial_index(routed))
        perm = A.wire_to_logical_perm(routed)
        r = np.empty_like(rho); r[:, perm[:, None], perm[None, :]] = rho
        P = N.basis_probs(r)
        assert abs(N.energy_from_freqs(P, co)[0] - e) < 1e-6
        # readout corruption + tensored inversion is exact at infinite shots
        Q = N.correct_readout(N.apply_readout(P, 0.03), 0.03)
        assert np.abs(Q - P).max() < 1e-12
    assert e - K.exact_ls(D, J)[0] < 10.0


def test_zne_rules():
    s = np.array([1.0, 1.5, 2.0])
    E = -20000 * np.exp(-0.05 * s)[None, :]
    e0, used = N.zne(E, c0=np.zeros(1))
    assert used[0] and abs(e0[0] + 20000) < 1e-6
    E2 = np.array([[-100.0, -120.0, -110.0]])   # non-monotone -> linear fallback
    e0, used = N.zne(E2, c0=np.zeros(1))
    assert not used[0] and abs(e0[0] - np.polyval(np.polyfit(s, E2[0], 1), 0)) < 1e-9


def test_pricing_formulas():
    # HQC = 5 + C (N1q + 10 N2q + 5 Nm)/5000 per circuit, N1q=12, Nm=5; mitigated = 3 bases x 3 scales
    exp = sum(5 + 1000 * (12 + 10 * n2 + 25) / 5000 for n2 in (24, 36, 48)) * 3
    assert abs(pricing.eval_cost("Helios", "mit", 1000) - exp) < 1e-9
    assert abs(pricing.eval_cost("Heron", "raw", 1000) - 3 * 1000 * 96 / 60 * 250e-6) < 1e-12
    assert abs(pricing.eval_cost("Forte", "raw", 1000) - 3 * (0.30 + 80.0)) < 1e-9
    assert pricing.eval_cost("Willow", "mit", 1000) is None


def test_rust_matches_python():
    exe = os.path.join(ROOT, "rust", "target", "release", "scospace")
    if not os.path.exists(exe):
        assert not REQUIRE_NATIVE, "Rust binary not built (cargo build --release)"
        print("   (Rust binary not built; skipped)"); return
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "space.csv")
        subprocess.run([exe, out, "0.24", "60", "4"], check=True, capture_output=True)
        data = np.genfromtxt(out, delimiter=",", names=True, dtype=None, encoding="utf-8")
    sp = L.Space(); cl = design.Classical(sp)
    assert len(data) == sp.n_complexes
    assert np.array_equal(data["scaffold"], sp.scaffold_idx) and np.array_equal(data["sixth"], sp.sixth_idx)
    assert all(a == sp.name(i) for i, a in enumerate(data["label"]))
    for col, ref in (("Delta", sp.Delta), ("B", sp.B), ("E_ls", cl.E_ls), ("E2", cl.E2), ("T_exact", cl.T_exact),
                     ("T_mf", cl.T_mf), ("T_pt2", cl.T_pt)):
        assert np.abs(data[col] - ref).max() < 1e-6 * max(1.0, np.abs(ref).max()), col


def test_qasm_roundtrip():
    D, J = 12500.0, 3300.0
    th, e = A.optimise_angles(K.hamiltonian_matrix(D, J))
    c0, terms = K.pauli_terms(D, J)
    entry = dict(c0=c0, terms=terms)
    for routed in (False, True):
        probs = {b: qasm.parse_and_simulate(qasm.circuit_qasm(th, routed, b, "t"))[0] for b in qasm.BASES}
        assert abs(qasm.energy_from_manifest_probs(probs, entry) - e) < 1e-6


def test_cuda_fallback():
    from .. import cuda_backend
    code, coef = A.noisy_program(False, 1.0)
    params = np.column_stack([np.zeros((2, 6)), np.full(2, 1e-3), np.full(2, 1e-2)])
    a = cuda_backend.run_batch(code, coef, params, A.initial_index(False))
    b = N.run_batch_numpy(code, coef, params, A.initial_index(False))
    assert np.abs(a - b).max() < 1e-12   # on this machine: NumPy fallback (CUDA path untested)


def test_rdm_gradient_and_halting():
    from ..rdm import RecurrentDepth
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(15, 4)), rng.normal(size=15)
    m = RecurrentDepth(4, 5, seed=1)
    _, g = m.loss_and_grad(X, y, 4, l2=1e-3)
    worst = 0.0
    for k, P in m.p.items():
        for i in np.ndindex(P.shape):
            o = P[i]; P[i] = o + 1e-6; lp, _ = m.loss_and_grad(X, y, 4, 1e-3)
            P[i] = o - 1e-6; lm, _ = m.loss_and_grad(X, y, 4, 1e-3); P[i] = o
            worst = max(worst, abs((lp - lm) / 2e-6 - g[k][i]))
    assert worst < 1e-6, worst
    m.xm, m.xs = np.zeros(4), np.ones(4)
    pred, depth = m.predict(X, tol=1e-3, T_max=64, backend="numpy")
    Y = m.iterates(X, 64, backend="numpy")
    assert np.all(depth >= 2) and np.all(depth <= 64)
    assert np.allclose(pred, Y[np.arange(len(X)), depth - 1] * m.ys + m.ym)


def test_cpp_stack_kernels():
    if native.ker_lib() is None:
        assert not REQUIRE_NATIVE, "C++ kernels not built (make -C cpp)"
        return
    from ..rdm import RecurrentDepth
    from .. import qaoa
    m = RecurrentDepth(7, 16, seed=3)
    X = np.random.default_rng(2).normal(size=(200, 7))
    assert np.abs(m.iterates(X, 30, "cpp") - m.iterates(X, 30, "numpy")).max() < 1e-12
    rng = np.random.default_rng(5)
    f = qaoa.problem(rng.random(12), rng.uniform(0.5, 1.5, 12), 4.0, 1.0)
    g, b = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
    pc = qaoa.probs(f, g, b, backend="cpp"); pn = np.abs(qaoa._numpy_state(f, g, b)) ** 2
    assert np.abs(pc - pn).max() < 1e-12
    params = rng.uniform(0, 1, (5, 2))
    assert np.abs(qaoa.batch_expect(f, params, 1, "cpp") - qaoa.batch_expect(f, params, 1, "numpy")).max() < 1e-10


def test_nasm_qaoa_cost():
    if native.par_lib() is None:
        assert not REQUIRE_NATIVE; return
    w = np.random.default_rng(4).random(12)
    assert np.abs(native.weighted_bitsum(12, w) - native.weighted_bitsum(12, w, force_numpy=True)).max() < 1e-12


def test_qaoa_ising_and_qasm():
    from .. import qaoa
    rng = np.random.default_rng(6)
    v, c = rng.random(12), rng.uniform(0.5, 1.5, 12)
    f = qaoa.problem(v, c, 4.0, 1.0)
    const, h, J = qaoa.ising(v, c, 4.0, 1.0)
    z = np.arange(4096)
    Z = np.array([1 - 2 * ((z >> (11 - q)) & 1) for q in range(12)], float)
    fi = const + h @ Z + np.einsum("ij,iz,jz->z", J, Z, Z)
    assert np.abs(fi - f).max() < 1e-9
    g, b = [0.3, 0.7], [0.4, 0.1]
    pq, ncx = qaoa.simulate_qasm(qaoa.circuit_qasm(h, J, g, b, "t"))
    assert ncx == 2 * 66 * 2
    assert np.abs(pq - qaoa.probs(f, g, b, backend="numpy")).max() < 1e-10


def test_rust_extended_scoring():
    from .. import genrl
    lab = genrl.rust_extended_labels()
    if lab is None:
        assert not REQUIRE_NATIVE, "Rust binary not built"; return
    ext = L.enumerate_extended_scaffolds()
    sp = L.Space(scaffolds=ext); cl = design.Classical(sp)
    sel = sp.pairs[:, 3] == C.SIXTH_SITE.index("CN")
    assert len(lab) == len(ext) == 161 and np.array_equal(lab, cl.ok_exact[sel])


def test_escalation_qlearning():
    """Toy loop: tiny penalty -> the learned policy stops at 1k shots; huge penalty -> escalates to 64k."""
    from types import SimpleNamespace
    from .. import genrl
    rng = np.random.default_rng(0)
    sp = L.Space(); cl = design.Classical(sp)
    disp = np.where((cl.T_exact > 250) & (cl.T_exact < 370))[0]   # near the thresholds
    n = len(disp)
    E = cl.E_ls[disp][:, None, None] + rng.normal(0, 1, (n, 8, 4)) * np.array([400, 200, 100, 1.0]) / design.k_per_cm()
    fake = SimpleNamespace(cl=cl, disp=disp, ev_disp={"Helios": {"mit": E}}, kpc=design.k_per_cm,
                           cfg=SimpleNamespace(shots=list(C.SHOTS), seed=1))
    r = genrl.escalation(fake, "Helios", lambdas=(1e-4, 1e3), episodes=4000)
    lo, hi = r["learned"]
    assert abs(lo["cost"] - r["fixed"][0]["cost"]) < 1e-9
    assert hi["cost"] > 0.95 * r["fixed"][-1]["cost"] and hi["err"] < 0.1   # (nearly) always escalates


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - fails}/{len(tests)} tests passed")
    sys.exit(1 if fails else 0)


main()
