#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'echo $? > validation_exit_status.txt' EXIT
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one currently idle GPU UUID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
.venv/bin/python -m assignment2.cartpole.validate diagnostics --run assignment2/results/cartpole/main_seed11
.venv/bin/python -m assignment2.cartpole.validate robustness --run assignment2/results/cartpole/main_seed11
.venv/bin/python -m assignment2.cartpole.validate suite
