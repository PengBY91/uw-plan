#!/usr/bin/env bash
# Single test entry point for both topics (review step 2, 2026-10-12): the uwbench suite (A detector, A1/A2 backend,
# B depth/map, audit, scoring) and the scripts' tests, all in .venv-eval, which now carries gtsam and opencv-headless
# next to the evaluation lock. Fails if any test is skipped, so a missing dependency cannot pass silently.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-eval/bin/python
log=$(mktemp)
trap 'rm -f "$log"' EXIT
PYTHONPATH=team/uwbench "$PY" -m unittest discover -s team/uwbench/tests "$@" 2>&1 | tee "$log"
# scripts/tests/ import the scripts under test by module name from scripts/data and scripts/openvins
PYTHONPATH=team/uwbench:scripts/data:scripts/openvins "$PY" -m unittest discover -s scripts/tests -p 'test_*.py' "$@" 2>&1 | tee -a "$log"
if grep -Eq 'skipped=[1-9]' "$log"; then
  echo "run_tests.sh: tests were skipped; the environment is incomplete" >&2
  exit 1
fi
