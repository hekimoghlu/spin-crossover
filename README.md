# Spin-crossover papers: independent reimplementation

| Folder | Paper | Languages |
|---|---|---|
| `method/` | Hybrid quantum-classical design loop (paper A) | Python (orchestration, training, figures), C++/OpenMP (CPU kernels: density-matrix, QAOA statevector, decoder forward pass), CUDA (GPU counterparts, NVIDIA only, untested), Rust (parallel enumeration and candidate scoring), x86-64 NASM (bit-level loops), OpenQASM 3 (hardware-ready circuits) |
| `materials/` | Nanoscale realisation (paper B) | Python (core), C++/OpenMP (structure factors, micrographs), Rust (hysteresis Monte Carlo), CUDA (untested) |
| `papers/` | Manuscript sources, read only by the validation step | LaTeX |

## Run

Requires Python 3 with numpy, scipy and matplotlib, plus g++ (OpenMP),
Rust/cargo (std only, no crates), and nasm.

```bash
cd method    && ./run_all.sh   # about 2 min on 4 cores
cd materials && ./run_all.sh   # about 1 min on 4 cores
```

Each run builds the native parts, runs the tests, runs the pipeline,
regenerates the figures in `figures/reimpl_fig*` and writes
`results/` and `VALIDATION.md`. If a native library is not built, the
NumPy path is used instead.

The CUDA kernels (with CuPy wrappers) have **not** been compiled or run;
they were written without a GPU. Only their NumPy fallbacks are tested.

## Agreement with the manuscripts

- `method/`: of 968 reported quantities, 269 match exactly, 540 are
  close (within 25 % or 5 units) and 157 do not match. All deterministic
  quantities match. The largest differences are in the calibrated
  surrogate (held-out error 28 K here vs 21 K), which changes the dispatch
  set, and in the QAOA dispatch test, where the probability of sampling
  the optimum is far lower here (0.001-0.003 vs 0.21-0.33) because the
  paper's objective constants are not fully stated.
- `materials/`: of 104 scalar quantities, 92 match, 6 are within 1 % and
  6 do not match. All 69 table cells match.

Every open modelling choice is listed in each folder's `ASSUMPTIONS.md`.
All components described in paper A are implemented, including the
recurrent-depth decoder and surrogate, the generative proposer, the
Q-learned shot-escalation policy and the twelve-qubit QAOA dispatch test.
The budget law and roughness ablation run in reduced form. CUDA targets
NVIDIA GPUs only; an NPU would need a different backend.
