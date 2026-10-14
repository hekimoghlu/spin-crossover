# scodesign: an independent reimplementation

> **This is an independent reimplementation written from the manuscript's method description
> ("Hybrid quantum-classical design of bistable spin-crossover materials ...", paperA.tex). It is
> not the code that generated the numbers in the manuscript.** Device evaluations are noisy
> density-matrix *simulations*, as in the manuscript. No hardware was used. The manuscript's
> reported numbers are read only by `scodesign/validation.py`, which prints them beside ours.

## Quick start

```bash
./run_all.sh          # build C++/NASM/Rust, run tests, pipeline, figures, QASM export, VALIDATION.md
./run_tests.sh        # build + tests only
python3 -m scodesign run --quick   # reduced smoke run
```

The full `run_all.sh` takes about **125 s** on 4 cores from a clean build, of which the pipeline is about 105 s (the implemented stack of sec. 6.3-6.7 adds about 65 s). It uses
the manuscript's settings throughout: 8 drift epochs, 16 for the calibration compound,
1k/4k/16k/64k shots per basis, and n_cal = 150. Nothing is subsampled.

## Language roles

| language | role in this package |
|---|---|
| Python (NumPy/SciPy) | orchestration, training loops (decoder, surrogate, Q-learning, QAOA parameter search), analysis, figures |
| C++ (OpenMP) | CPU compute kernels: noisy density-matrix batches (`dmsim.cpp`), 12-qubit QAOA statevector (`qaoa_expect`, `qaoa_batch`), recurrent-depth forward pass (`rdm_forward`) in `kernels.cpp`; ctypes with NumPy fallback |
| CUDA (NVIDIA only) | GPU counterparts (density-matrix batch, batched QAOA evolution, batched decoder/surrogate inference) via CuPy `RawKernel`, falling back to C++/NumPy. **Untested: no GPU or nvcc here.** |
| Rust (std threads) | parallel search and enumeration: whole-space sweep, and generating plus exactly scoring the generative proposer's four- and five-ligand candidates |
| x86-64 NASM | bit-level hot loops: Pauli parity expectations, popcount basis enumeration, QAOA bitstring cost table (`weighted_bitsum`) |
| OpenQASM 3 | hardware-ready circuits: ansatz and twin (native and routed) and the QAOA circuits |

An NPU would need a different backend; none is provided.

## Layout

| path | content |
|---|---|
| `scodesign/constants.py` | inputs stated in the paper: ligand table, platform table, prices, design condition |
| `scodesign/ligands.py` | ligand-space enumeration (301 / 1505 / 1204) and Jørgensen ligand field |
| `scodesign/kanamori.py` | Kanamori d6 model, hard-core-boson block, 35-term Pauli decomposition (3 groups), exact LS, MF, PT2, full ten-mode cross-check |
| `scodesign/design.py` | T1/2, straddle condition, recalibration fit, leverage standard error |
| `scodesign/ansatz.py` | Givens ansatz, angle optimisation, 4-CNOT/2-Ry compilation, linear-chain SWAP routing, CNOT folding, op-list programs |
| `scodesign/noise.py` | batched density-matrix kernel (NumPy), Z/X/Y basis probabilities, readout flip and tensored correction, multinomial shots, energy from histograms, exponential/linear ZNE |
| `scodesign/device.py` | device evaluations: ansatz + twin, drift epochs, four arms (raw, mitigated, reference-scaled, paired-mitigated) |
| `scodesign/pricing.py` | Braket / IBM / HQC price models |
| `scodesign/loop.py` | the full pipeline: router predictions, calibration, band, dispatch, policies, routing, tables, budget law, roughness ablation, eps sweep, sensitivity, scaling |
| `scodesign/qasm.py` | OpenQASM 3 export (native + routed, ansatz + twin, Z/X/Y) + manifest + independent re-simulation |
| `scodesign/figures.py` | `figures/reimpl_fig{1,3,4,5,6a,7}.{pdf,png}` |
| `scodesign/rdm.py` | recurrent-depth network (NumPy BPTT + Adam, adaptive halting) |
| `scodesign/stack.py` | recurrent-depth decoder trained on twin circuits; recurrent-depth surrogate study |
| `scodesign/genrl.py` | generative proposer (site-composition prior + logistic scorer); Q-learned shot escalation |
| `scodesign/qaoa.py` | twelve-qubit QAOA dispatch test, Ising mapping, QASM 3 export and parser |
| `scodesign/validation.py` | writes `VALIDATION.md` |
| `scodesign/native.py` | ctypes bindings (C++, NASM) with NumPy fallback |
| `scodesign/cuda_backend.py`, `cuda/dmsim_kernel.cu` | CUDA kernel and CuPy wrapper (**untested**) |
| `scodesign/tests/` | plain-assert tests: `python3 -m scodesign.tests` |
| `cpp/` | C++17/OpenMP batched density-matrix kernel (`libdmsim.so`) |
| `asm/` | x86-64 NASM parity-expectation and popcount-enumeration routines (`libparity.so`) |
| `rust/` | std-only Rust whole-space sweep (`scospace`) |
| `results/results.json` | every computed number (and `space_rust.csv`) |
| `qasm/` | 600 `.qasm` files, `manifest.json`, `verification.json` |

## Language components and test status

* **Python (NumPy/SciPy)**: the core, with fixed seeds. 13/13 tests pass.
* **C++ (OpenMP)**: `dmsim_run_batch` applies a shared gate program (Ry, CNOT, 1q/2q
  depolarising) to a batch of tasks, each with its own angles and noise strengths, on real 32×32
  density matrices. **The pipeline uses it for every device evaluation.** It is called through
  ctypes, and NumPy takes over if the library is not built (`SCODESIGN_NO_NATIVE=1` forces the
  fallback). The test checks it against the NumPy kernel to < 1e-9 on native and routed circuits
  at all three folding scales; the observed difference is about 1e-16. **Builds and passes.**
* **Rust (std only, `std::thread`)**: `scospace` enumerates the whole space and computes the
  ligand field, the mean-field gap, PT2 and the exact LS energy (cyclic Jacobi on the 10×10
  block), then writes CSV. The test checks it against the Python Δ, B, E_LS, E2 and T values to
  1e-6 relative, and checks labels and order exactly. **Builds offline and passes.**
* **x86-64 NASM**: `parity_expect` (POPCNT-based Pauli-Z parity expectations from histograms)
  computes every energy estimate in the pipeline. `enum_popcount` enumerates the seniority-zero
  basis. The test compares both with NumPy (< 1e-12). **Builds and passes.**
* **CUDA**: `cuda/dmsim_kernel.cu` with a CuPy `RawKernel` wrapper. **It cannot be compiled or
  run here because there is no GPU and no nvcc, and it has never been tested.** The wrapper falls
  back to NumPy when CuPy or a GPU is absent. The only test exercises that fallback.
* **C++ stack kernels** (`cpp/kernels.cpp`): the QAOA statevector runs every QAOA objective
  evaluation and the p = 1 landscape. `rdm_forward` runs every decoder and surrogate inference.
  Tests check both against NumPy to < 1e-10. **Builds and passes.**
* **NASM `weighted_bitsum`** builds the QAOA cost table for all 4096 bitstrings. The test checks
  it against NumPy. **Passes.**
* **Rust `scospace --extended`** generates the 161 four- and five-type candidates and scores
  them exactly with std threads. The pipeline uses its labels, and the test checks them against
  the Python solver. **Passes.**
* **OpenQASM 3**: 50 dispatched complexes × {ansatz, twin} × {native, routed} × {Z, X, Y} = 600
  files. They were checked by re-simulating every file with an independent parser and rebuilding
  the energy from the manifest coefficients (max deviation 5e-11 cm^-1). Structural checks
  confirm 24 / 48 CNOTs and nearest-neighbour-only CNOTs in the routed files. (The manuscript
  ships OpenQASM 2.0.) The QAOA circuits for p = 1, 2, 3 are in `qasm/qaoa/`, with 132 CNOTs per
  layer. They are re-simulated by an independent parser, with max probability deviation
  < 1e-9 (asserted in the run).

## Validation summary (details in VALIDATION.md)

* **Deterministic parts reproduce exactly.** These are the counts (301/1505/1204), 28 % low-spin,
  141 / 20 / 211 successful designs (exact/MF/PT2), and agreement of 90.0 % and 94.2 %. They also
  include the MF/PT2 RMS and bias (392/-389 K, 178/+175 K), the crossovers 17.9 / 19.3, the Δ and B
  ranges, the correlation-energy range, the calibration compound (530 cm^-1, 106 K), the
  139 cyanide / 2 azide split, and the whole k × dS sensitivity table. Gate counts, prices, the
  scaling table and the 161/54 extended-space labels match as well.
* **Noise propagation is close.** For example, raw bias on Helios is -70 K against -70 to -71 K
  in the paper, Heron is -423 to -428 K against -427 to -430 K, and Cepheus is about -1675 K
  against -1647 K. Mitigated RMS on Helios at 64k is 14.0 K against 14 K. The reference-state
  table (tab_rsem) and the roughness table (tab_rough) agree within a few kelvin for most cells.
* **Router.** It picks IBM Heron r3, mitigated arm, 1,000 shots as the calibration arm, the same
  arm as the paper.
* **Mismatches.** The deployed fit is a single noise draw. Its held-out RMS is 28 K against 21 K,
  max 108 K against 65 K, and the fit coefficients are a = -1174 cm^-1, b = 0.244 against -817 and 0.371. As a result the band is 45 K (paper
  51 K) and 50 complexes are dispatched (paper 53). Every count over the dispatched pairs shifts
  with this. With our draw, 11 of the 50 dispatched pairs are successful designs, against 17 of
  53 in the paper, so all-wrong raw policies score 11 rather than 17. The routed policy sends 47
  candidates to Helios mitigated 64k and 3 to Heron mitigated 64k (paper: all 53 to Helios), with
  5 wrong decisions in both. The calibration-compound predictions for Cepheus raw are about 25 %
  larger than the paper's. The residual mitigated bias on Heron and Willow is 5-12 K smaller than
  the paper's. The absorbed fraction of the hidden roughness term is 22 % against 36 %.

## Implemented stack (sec. 6.3-6.7): key results (simulation)

* **Recurrent-depth decoder.** It is trained only on twin circuits. It beats the paired-mitigated
  arm in 18 of 20 platform and shot cells (paper: 17), extrapolation in 13 (paper: 13) and
  reference scaling in 2 (paper: 1). Mean depth is 4.7-8.4 iterations per cell (paper: 7.5-15.4).
* **Recurrent-depth surrogate.**

  | label source | ours: line | ours: ensemble of five | paper: line | paper: ensemble |
  |---|---|---|---|---|
  | device-extrapolated | 4 K | 59 K | 34 K | 62 K |
  | decoder labels | 16 K | 53 K | 17 K | 58 K |
  | exact | 2.9 K | 1.0 K | 3 K | 6 K |

  The depth-|E_corr| correlation is +0.02 (paper: -0.42).
* **Generative proposer.**

  | quantity | ours | paper |
  |---|---|---|
  | prior hit rate, all draws | 87 % | 87 % |
  | prior hit rate, distinct proposals | 35 % | 36 % |
  | distinct scaffolds reached | 153 | 150 |
  | scorer precision at top 10 / 20 / 40 | 100 / 100 / 95 % | 100 / 100 / 95 % |
  | best extended scaffold | 1py+1NH3+1en+2bipy | 1py+1NH3+1en+2bipy |

* **Q-learned escalation.** On Helios at λ = 1 it costs 14,568 HQC with error 0.27, against a
  fixed 16k policy at 15,142 HQC with error 0.29 (paper: 12,336 at 0.335 against 15,142 at
  0.344). Saving at the top of the frontier: Helios 8 % (paper 1 %), Heron 0.5 % (paper 7 %).
* **QAOA.**

  | quantity | ours | paper |
  |---|---|---|
  | approximation ratio, p = 1 / p = 3 | 0.82 / 0.94 | 0.956 / 0.966 |
  | P(optimal), p = 1 / p = 3 | 0.001 / 0.003 | 0.21 / 0.33 |
  | evaluations at p = 3 | 4,745 | 7,063 |
  | CNOTs at p = 3 | 396 | 396 |
  | brute-force evaluations | 4,096 | 4,096 |

  The large differences in P(optimal) and the ratio follow from the unstated objective constants
  (ASSUMPTIONS.md §33).

## Not reimplemented

* The materials-companion numbers (`nb*`, `ml*`, `mg*`) and a few loosely defined macros, such as
  `nWindowThreeHundred`, `nearestBelowFever` and the prose-only depth sentences.
* The QASM hardware cost estimates (`qasmCost*`).
* The budget law and the roughness ablation are reimplemented in reduced form (ASSUMPTIONS.md
  §24-25). The roughness decision-level analysis uses fewer replicates.
* The recurrent-depth models, the generative proposer, the escalation policy and QAOA are
  implemented, but with our own architectures and constants (ASSUMPTIONS.md §28-33).

See `ASSUMPTIONS.md` for every choice made where the manuscript is silent.
