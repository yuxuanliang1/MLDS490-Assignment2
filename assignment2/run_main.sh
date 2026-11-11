#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'echo $? > main_exit_status.txt' EXIT
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one currently idle GPU UUID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
PY=.venv/bin/python
RUN=assignment2/results/cartpole/main_seed11
$PY -m pytest assignment2/tests -q > tests_main.log 2>&1
$PY -m assignment2.cartpole.train --output "$RUN" --seed 11 --episodes 1000 --max-steps 80000
$PY -m assignment2.cartpole.validate finalize --run "$RUN"
