#!/usr/bin/env bash
# Build native components and run the plain-assert test suite (+ cargo's own test target check).
set -euo pipefail
cd "$(dirname "$0")"
make all
python3 -m scodesign.tests
