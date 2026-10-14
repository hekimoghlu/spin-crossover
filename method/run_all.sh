#!/usr/bin/env bash
# Build everything, run tests, run the full pipeline, figures, QASM export and validation.
set -euo pipefail
cd "$(dirname "$0")"
start=$(date +%s)
make all
python3 -m scodesign.tests
python3 -m scodesign all "$@"
./rust/target/release/scospace results/space_rust.csv 0.24 60 4
end=$(date +%s)
echo "run_all.sh total wall time: $((end - start)) s" | tee results/run_all_time.txt
