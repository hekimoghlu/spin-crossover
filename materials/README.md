# scmaterials: reimplementation of the computational methods of the spin-crossover materials paper

> **This is an independent reimplementation, written from the manuscript's method
> description. It is not the code that generated the numbers in the manuscript.**
> It was written from the text of *"Nanoscale realisation of a quantum-designed
> spin-crossover analyte sensor"* (paper B) and the model sections of its companion
> method paper (paper A: problem formulation, ligand-field surrogate, Kanamori model).
> Where the manuscript is underspecified, the choices made are listed in
> `ASSUMPTIONS.md`. The comparison with every reported number, mismatches included,
> is in `VALIDATION.md`.
>
> **The CUDA component (`cuda/`, `scmaterials/cuda_kernels.py`) has never been compiled
> or run.** The development environment had no GPU, nvcc or CuPy. It is untested and
> is not used for any result.

## Quick start

```bash
./run_all.sh          # builds C++ and Rust, runs all tests, regenerates results, figures and VALIDATION.md (~70 s on 4 cores)
python3 tests/run_tests.py   # tests only (needs cpp/build and rust/target built)
```

Requirements: Python 3 with NumPy, SciPy and Matplotlib; g++ with OpenMP; and Rust
(cargo, std only, no crates). `run_all.sh` reads the manuscript at
`../papers/paperB.tex` for the comparison step only; override the
path with `PAPER_TEX=...`.

## Layout

| path | content |
|---|---|
| `scmaterials/ligandfield.py` | Jørgensen ligand-field surrogate; exact 5-orbital Kanamori model (10-dim seniority-zero block, closed-form HS determinant, plus a 210-dim full Fock-space check); enumeration of the 301 scaffolds and 1505 complexes; T½ and the straddle condition |
| `scmaterials/design.py` | design set (141), interferent/selectivity map against six physiological donors, f\* and margins, cross-analyte silence, temperature-window statistics, competition budget |
| `scmaterials/sizemap.py` | Slichter–Drickamer mean-field loop width against diameter, critical diameter d_c, quasi-static sweep |
| `scmaterials/geometry.py` | Fe–N coordination sphere, rigid-linkage Hofmann Fe(pz)[Pt(CN)₄] cell and atom coordinates |
| `scmaterials/readout.py` | Curie moment, NV stray-field contrast and SNR times, array-pitch rule, response-time budget, graphene interface budget |
| `scmaterials/radiation.py` | ionisations per gray, Bethe proton stopping power, Rutherford + NRT displacement count |
| `scmaterials/scattering.py` | electron scattering factors, kinematic structure factors, projected potential ([001] and [100]), radial diffraction profile with Scherrer broadening, resolvability and fringe counting |
| `scmaterials/micrograph.py` | simulated bright-field micrographs of particle ensembles with size distributions; lattice image of one particle with its FFT |
| `scmaterials/kernels.py` | backend dispatch: C++/OpenMP via ctypes, NumPy fallback, CUDA (untested, opt-in only) |
| `scmaterials/mc_ref.py` | pure-Python reference for the Rust Monte Carlo (same RNG stream), plus Rust wrappers |
| `scmaterials/pipeline.py` | computes everything into one dictionary |
| `cpp/scmk.cpp`, `cpp/Makefile` | OpenMP kernels: `sf_sum` (structure factor over atoms × g-vectors), `synth_image` (Fourier synthesis of projected potentials), `render_spheres` (projected-thickness rendering of micrographs) |
| `rust/` | `scmc` (std only, `std::thread`): Slichter–Drickamer loop widths, and Ising-like Monte Carlo hysteresis loops of particle ensembles against size, written as CSV |
| `cuda/scmk_kernels.cu`, `scmaterials/cuda_kernels.py` | CUDA versions of `sf_sum` and `synth_image` behind a CuPy RawKernel wrapper. **UNTESTED; never compiled.** |
| `scripts/compute_results.py` | writes `results/results.json` |
| `scripts/make_figures.py` | writes `figures/reimpl_fig1..5.{pdf,png}` (analogues of paper figures 1–4, plus a Rust Monte Carlo figure); every simulated image is labelled SIMULATED |
| `scripts/validate.py` | writes `VALIDATION.md` (paper value against reimplementation value, with a verdict) |
| `tests/run_tests.py` | plain-script test runner |

## Component and test status (as run here)

- **Python package:** 13/13 tests pass. The tests cover the model check (seniority-zero
  block against the full Fock space to 1e-8 cm⁻¹), closed forms, geometry, Bethe
  against PSTAR, and the size-map consistency.
- **C++/OpenMP:** builds with `make`. All three kernels agree with NumPy to rtol 1e-12,
  and so does the whole projected-potential pipeline (max deviation < 1e-12). The
  kernels are used by default for structure factors, potentials and micrographs.
- **Rust:** `cargo build`, and `cargo test` passes 4 unit tests. The Python test checks
  the Rust mean-field loop widths against SciPy to < 1e-6 K, and the Rust Monte Carlo
  trajectories against the Python reference bit for bit (same SplitMix64 stream).
- **CUDA:** not compiled and not run. It is never selected implicitly; a test checks
  this.

## What is not reimplemented

The noisy quantum-device simulation, the device-calibrated surrogate and the routing of
the companion paper are not reimplemented. The paper-B macros that quote those results
(`agreePTc`, `rmsPTc`, `nCal`) are listed as not reimplemented. The design set used
here is the one from the exact solver, which is the paper's own design definition: 141
designs, identical to the paper's count. The technology-stack assessment, the roadmap
and the predictions are qualitative and have no computation behind them.

## Determinism

Every stochastic element uses a fixed seed: NumPy `default_rng` seeds in `micrograph.py`
and the SplitMix64 seeds in the Rust Monte Carlo.
