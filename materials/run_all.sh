#!/usr/bin/env bash
# Build every component, run every test, regenerate results, figures and VALIDATION.md.
# The CUDA component is NOT built or run (no GPU / nvcc in the development environment).
set -euo pipefail
cd "$(dirname "$0")"
START=$(date +%s)
PAPER_TEX="${PAPER_TEX:-../papers/paperB.tex}"

echo "== [1/7] C++/OpenMP kernel library"
make -C cpp

echo "== [2/7] Rust (std only): build + unit tests"
(cd rust && cargo build --release --offline && cargo test --release --offline)

echo "== [3/7] Python tests (incl. C++ vs NumPy, Rust vs Python)"
python3 tests/run_tests.py

echo "== [4/7] Results JSON"
python3 scripts/compute_results.py

echo "== [5/7] Rust mean-field + Monte Carlo CSVs (4 threads)"
./rust/target/release/scmc mf results/rust_mf.csv
./rust/target/release/scmc mc results/rust_mc.csv 100 4

echo "== [6/7] Figures"
python3 scripts/make_figures.py

echo "== [7/7] Validation against the manuscript"
python3 scripts/validate.py "$PAPER_TEX"

echo "CUDA: cuda/scmk_kernels.cu was NOT compiled or run (untested; no GPU here)."
echo "run_all.sh finished in $(( $(date +%s) - START )) s"
