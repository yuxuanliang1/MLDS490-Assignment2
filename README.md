# MLDS 490 — Assignment 2: Deep Q-Networks

**[Read the Final Report](MLDS_490_Assignment_2.pdf)**

Original PyTorch implementations and measured experiments for Northwestern
University's MLDS 490: **CartPole-v1** and **MsPacman-v5**.

## Project Overview

Both agents implement Classic DQN with a neural action-value function, randomized
experience replay, an independent target network, epsilon-greedy exploration,
Huber loss, Adam, and gradient clipping. The objectives are 500-step CartPole
balance and improved undiscounted MsPacman score. No external RL framework is used.

## Final Report

[Read the Final Report](MLDS_490_Assignment_2.pdf) for mathematical formulation,
figures, scientific validation, negative results, and the three-month research plan.
The supplied 22-page PDF is preserved unchanged. Its title page identifies this
repository; duplicated metadata placeholders remain in Section 7.

## Results Summary

| Measurement | CartPole-v1 | MsPacman-v5 |
|---|---:|---:|
| Discount factor | 0.95 | 0.99 |
| Selected training seed | 33 | 41 |
| Completed training episodes | 942 | 3,114 |
| Agent decision steps | 80,014 | 2,000,528 |
| Held-out greedy episodes | 500 | 500 |
| Mean undiscounted return / score | 499.266 | 1,501.06 |
| Population standard deviation (`ddof=0`) | 11.731634 | 493.720849 |
| Perfect 500-step episodes | 498/500 (99.6%) | Not applicable |
| First / final 100 training mean | 22.83 / 177.32 | 272.1 / 994.2 |

Raw evaluation episodes and summaries: [CartPole](assignment2/results/cartpole/seed33/evaluation500/)
and [MsPacman](assignment2/results/mspacman/main_seed41/evaluation500/).
These are separate from exploratory training rewards and checkpoint-selection
validation. The two games' reward units are not directly comparable.

## Methods

CartPole uses float32 observations `(x, x_dot, theta, theta_dot)` and a
**4–128–128–2 MLP** with ReLU hidden layers. MsPacman uses **88×80 single-frame
int8 images**, three ReLU convolutions (32/64/64 channels, kernels 8/4/3,
strides 4/2/1), a 512-unit ReLU layer, and nine linear Q outputs. There is no softmax.

| Setting | CartPole | MsPacman |
|---|---|---|
| Adam learning rate | `5e-4` | `1e-4` |
| Replay capacity / warmup | 50,000 / 1,000 | 50,000 / 10,000 |
| Minibatch | 128 | 32 |
| Optimize every | 1 decision | 4 decisions |
| Hard target copy every | 500 optimizer updates | 5,000 optimizer updates |
| Epsilon schedule | 1 → .05 over 20,000 decisions | 1 → .1 over 500,000 decisions |
| Loss / gradient norm cap | Huber / 10 | Huber / 10 |

Episodes end on `terminated` or `truncated`; only genuine termination masks
the bootstrap target. Final-transition rewards and both flags are preserved.
The [saved configurations](assignment2/PUBLIC_RESULTS.md) define the measured
budgets; current CLI defaults have larger step/time limits.

## Repository Structure

```text
MLDS490-Assignment2/
├── README.md
├── MLDS_490_Assignment_2.pdf
├── requirements.txt
├── PUBLICATION_MANIFEST.json
└── assignment2/
    ├── cartpole/                 # DQN, replay, train/evaluate/plot/validate
    ├── mspacman/                 # CNN, Atari setup, replay, recovery, experiments
    ├── tests/                    # Focused CPU correctness tests
    ├── results/{cartpole,mspacman}/ # Selected original metrics and PNG figures
    ├── README.md                 # Detailed commands and experiment protocols
    ├── PUBLIC_RESULTS.md         # Included-data and omitted-artifact guide
    ├── verify_public_results.py  # Read-only verification without model weights
    ├── requirements*.txt
    └── launch*.py, run_*.sh       # Sequential execution utilities
```

## Installation and Dependencies

The measured Linux environment used **Python 3.12.6**, **PyTorch 2.14.0 / CUDA
13.0**, Gymnasium 1.4.0, ALE 0.10.2, NumPy 2.5.3, Matplotlib 3.11.2,
pandas 3.0.5, pytest 9.2.0, and psutil 7.2.2. Versions are pinned in the dependency files.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use an official PyTorch build compatible with the driver and GPU architecture;
see [PyTorch installation](https://pytorch.org/get-started/locally/). The recorded
Blackwell GPU requires `sm_120` support. Pins describe the measured environment,
not a promise that every wheel is available on every platform. No torchvision,
torchaudio, Stable-Baselines3, or AutoROM dependency is needed by this code.

The working ALE installation already supplied the MsPacman ROM. Check availability
with `python -c "from ale_py import roms; print(roms.get_rom_path('ms_pacman').name)"`.
If absent, follow the [official ALE installation instructions](https://ale.farama.org/getting-started/)
for the installed release. ROM binaries and course materials are not distributed here.

Training/evaluation CLIs intentionally require **deepdish5 and exactly one NVIDIA
GPU**; they do not silently fall back to CPU. CPU unit tests and saved-result
verification do not require GPU access. Before any substantive run on deepdish5:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
# Set GPU_UUID to one currently idle UUID from that inventory.
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
```

Set visibility before importing torch. Run experiments sequentially. The runtime
audit checks one usable CUDA device and a real matrix product with known sum 3,680.

## Reproducing CartPole

Run from the repository root with the environment activated. Use a **fresh** path;
the published evidence directories already exist and training refuses to overwrite them.

```bash
CP=assignment2/results/reproduce_cartpole/seed33
python -m assignment2.cartpole.train --output "$CP" \
  --seed 33 --episodes 1000 --max-steps 80000
python -m assignment2.cartpole.validate finalize --run "$CP"
```

`finalize` reads the selected checkpoint, runs exactly 500 greedy episodes, and
generates figures. Alternatively, replace it with the individual commands below.
`CHECKPOINT` is the path named by the new run's `metrics/selection.json`; the
recorded seed-33 experiment selected `checkpoints/best_ep0942.pt`.

```bash
python -m assignment2.cartpole.evaluate --checkpoint "$CHECKPOINT" --output "$CP/evaluation500"
python -m assignment2.cartpole.plots --run "$CP" --checkpoint "$CHECKPOINT" \
  --evaluation "$CP/evaluation500" --device cuda:0
CUDA_VISIBLE_DEVICES=-1 python -m pytest assignment2/tests/test_cartpole_dqn.py -q
```

## Reproducing MsPacman

`MsPacman-v5` is an exact registration alias of installed `ALE/MsPacman-v5`, with
nine actions, native frameskip 4, sticky-action probability .25, and 108,000 raw
frames per game. There is no additional repeat wrapper, frame stacking, reward
clipping, random no-op reset, or episodic-life termination. Pixel preprocessing
follows the assignment; see [environment.py](assignment2/mspacman/environment.py)
and [preprocessing.py](assignment2/mspacman/preprocessing.py).

```bash
python -m assignment2.mspacman.probe --output assignment2/results/reproduce_atari_preflight
MS=assignment2/results/reproduce_mspacman/main_seed41
python -m assignment2.mspacman.train --output "$MS" --seed 41 \
  --episodes 4000 --max-steps 2000000 --max-hours 4
python -m assignment2.mspacman.validate finalize --run "$MS"
```

Alternatively, replace `finalize` with individual evaluation/plot commands:

```bash
python -m assignment2.mspacman.evaluate --checkpoint "$CHECKPOINT" --output "$MS/evaluation500"
python -m assignment2.mspacman.plotting --run "$MS" --device cuda:0
CUDA_VISIBLE_DEVICES=-1 python -m pytest assignment2/tests/test_mspacman.py -q
```

Set `CHECKPOINT` from this run's selection JSON; the recorded model was
`checkpoints/best_ep03114.pt`. `finalize` also performs learning-stage and Bellman
diagnostics. Do not execute both routes into the same evaluation directory.

## Experimental Validation

- **CartPole:** seeds 11/22/33; hard/soft/every-update targets; random versus
  correlated recent replay; gamma .90/.95/.99; ten-step and independent NumPy
  target-network diagnostics; 405 altered-initial-state episodes.
- **MsPacman:** six approximately 100,000-decision runs examine seeds 41/52/63,
  target lag, recent replay, and slower epsilon decay. Each selected short-run
  model has **30 diagnostic episodes**, not a full-budget 500-game replication.
- Both games test terminal masking, replay wraparound/underflow, epsilon extremes,
  action indexing, frozen targets, numerical guards, and checkpoint consistency.
- Atari scalar Bellman arithmetic agrees within `8.55e-6`; CartPole's independent
  target MLP agrees within `3.02e-6`. These checks do not reconstruct historical targets.
- Atari recovery/rollback smoke matches parameters exactly; its repeated seed-41
  prefix matches 100,031 recorded transitions and saved episode-100 parameters.
  Exact recovery is bounded to the tested hardware/software configuration.

Publication checks passed all **30 focused tests on CPU** with CUDA hidden;
source hashes matched the local algorithm and test files. No new training was run.

[CartPole comparisons](assignment2/results/cartpole/comparison.csv) and
[MsPacman comparisons](assignment2/results/mspacman/validation/comparison.csv)
include negative/inconclusive outcomes. The recent-replay Atari score-difference
interval includes zero; it does not establish an improvement. Full commands and
protocol details are in the [technical README](assignment2/README.md).

Verify published numerical claims **without training or checkpoints**:

```bash
python -m assignment2.verify_public_results
```

## Reproducibility Notes

Training ran sequentially on one **RTX 5060 Ti 16 GB** on deepdish5, with CUDA 13.0
and driver 615.71.09. CartPole seed 33 took about 180 seconds; MsPacman seed 41
took 54.48 minutes plus 4.19 minutes for final evaluation. Timing is machine-specific.
Atari completed 7,992,770 measured raw frames; a decision advances up to four
frames, and optimizer updates occur only every fourth decision after warmup.

CartPole limits were 1,000 episodes / 80,000 decisions; Atari limits were 4,000
games / 2 million decisions / 4 hours. Caps are checked at episode boundaries.
Validation and test reset panels are separate; seeds and selected configurations
are included. Cross-device/version bitwise reproducibility is not guaranteed.

**Model weights are not included.** Exact selected-policy inference requires
separately obtaining those checkpoints; training anew can select different weights.
Full Atari recovery additionally requires replay sidecars and RNG/reference state.
The public data support verification of reported scores, not checkpoint-free inference.

## Limitations

CartPole's strong best run is seed-sensitive and retains two standard-reset
failures; altered-state success is 293/405. Only one Atari seed has the full main
budget. Atari stopped at its decision cap after 3,114 games, below the suggested
4,000–6,000-game scale. Validation fluctuates, small reused selection panels can
overfit, and single-frame Atari observations hide motion. These experiments
demonstrate learning, not universal convergence or optimality.

## Author

**Yuxuan Liang** · [GitHub: yuxuanliang1](https://github.com/yuxuanliang1)
