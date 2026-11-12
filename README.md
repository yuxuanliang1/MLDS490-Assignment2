# MLDS 490 Assignment 2: Deep Q-Networks

Classic DQN implementations and experiments for **Northwestern University MLDS 490**,
by **Yuxuan Liang**. CartPole-v1 uses a 4-128-128-2 MLP with gamma 0.95;
MsPacman-v5 uses an 88 x 80 single-frame CNN with gamma 0.99. Both use experience
replay, a separate target network, epsilon-greedy exploration, Adam, Huber loss,
and gradient clipping.

**[Final Report](MLDS_490_Assignment_2.pdf)** for the methods, eight principal
figures, controlled validation experiments, negative findings, and three-month
research plan.

## Results

Each selected policy was evaluated greedily on **exactly 500 separate episodes**.
Standard deviations below are population SD (`ddof=0`) across individual returns.

| Measurement | CartPole-v1 | MsPacman-v5 |
|---|---:|---:|
| Training seed | 33 | 41 |
| Completed training episodes / games | 942 | 3,114 |
| Agent decisions | 80,014 | 2,000,528 |
| Held-out evaluation episodes | 500 | 500 |
| Mean evaluation return / raw score | 499.266 | 1,501.06 |
| Population SD | 11.73 | 493.72 |
| Median; range | 500; 287-500 | 1,430; 590-4,800 |
| Perfect 500-step episodes | 498/500 (99.6%) | Not applicable |
| First / last 100 exploratory training mean | 22.83 / 177.32 | 272.1 / 994.2 |

Individual evaluation records: [CartPole](assignment2/results/cartpole/seed33/evaluation500/)
and [MsPacman](assignment2/results/mspacman/main_seed41/evaluation500/). Evaluation,
exploratory training, and checkpoint-selection validation are distinct panels;
the games' reward units are different. MsPacman advanced 7,992,770 measured raw
frames; decisions, frames, and optimizer updates are counted separately.

## Repository and data availability

```text
MLDS490-Assignment2/
|-- MLDS_490_Assignment_2.pdf     # Compiled academic report
|-- requirements.txt           # Includes package dependency files
|-- PUBLICATION_MANIFEST.json   # Original evidence hashes and sizes
`-- assignment2/
    |-- cartpole/               # MLP DQN, replay, training, evaluation, plots
    |-- mspacman/               # CNN DQN, Atari setup, diagnostics, recovery
    |-- tests/                  # Focused correctness checks for both parts
    |-- results/                # Selected original CSV/JSON/PNG evidence
    |-- verify_public_results.py
    |-- PUBLIC_RESULTS.md       # Included and omitted evidence
    `-- README.md               # Detailed reproduction protocols
```

The public repository contains implementation/tests, selected metrics and figures,
and small Atari reference/diagnostic image arrays. These reproduce score summaries,
training curves, and comparison tables from saved measurements. Editable LaTeX,
course handouts, full step/update logs, replay archives, and other model weights remain
local. See [data availability](assignment2/PUBLIC_RESULTS.md).

**The two selected inference checkpoints are included:**

- [CartPole seed 33, episode 942](assignment2/results/cartpole/seed33/checkpoints/best_ep0942.pt) (144,455 bytes).
- [MsPacman seed 41, episode 3,114](assignment2/results/mspacman/main_seed41/checkpoints/best_ep03114.pt) (11,633,449 bytes).

They contain online/target networks and configuration metadata, not complete
resumable training state. Optimizer/replay/RNG recovery archives remain local.
Retraining can select different weights, and bitwise repetition across hardware
or software versions is not guaranteed.

## Installation and environment checks

Run from the repository root. The recorded Linux environment used **Python 3.12.6**,
PyTorch **2.14.0 / CUDA 13.0**, Gymnasium 1.4.0, ALE 0.10.2, NumPy 2.5.3,
Matplotlib 3.11.2, pandas 3.0.5, pytest 9.2.0, and psutil 7.2.2.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The root dependency file includes `assignment2/requirements_mspacman.txt`, which
includes the shared `assignment2/requirements.txt`. On Windows the activation
command is `.venv\Scripts\Activate.ps1`; substantive runs enforce Linux deepdish5.

The torch version pin does **not** select its CUDA wheel flavor. On Blackwell,
first install/provide a torch 2.14.0 build supporting `sm_120`, using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then install
the requirements. A standard pip install alone is not evidence of the recorded
CUDA build; wheel availability depends on platform and package index. Inspect it:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

ALE's recorded installation supplied the MsPacman ROM. If ROM discovery fails,
follow the [official ALE setup](https://ale.farama.org/getting-started/) for the
installed release. No ROM binaries are distributed, and no AutoROM step is assumed.
`make_env()` registers and checks `MsPacman-v5` against `ALE/MsPacman-v5`. This
small **CPU-only smoke** checks the ROM, both environments, and one action:

```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import gymnasium as gym
from ale_py import roms
from assignment2.mspacman.environment import make_env
assert roms.get_rom_path("ms_pacman").is_file()
for env in (gym.make("CartPole-v1"), make_env()):
    try:
        obs, _ = env.reset(seed=0)
        env.step(0)
        print(env.spec.id, obs.shape, env.action_space.n)
    finally:
        env.close()
PY
```

Atari uses nine actions, native frameskip 4, sticky-action probability 0.25, and
a 108,000-raw-frame limit. There is no extra action repeat, frame stacking, reward
clipping, random no-op reset, or episodic-life termination. Both games stop on
either end flag and bootstrap only when `terminated` is false.

## Training and evaluation

The experiment CLIs require **deepdish5 and exactly one usable NVIDIA GPU**.
Select one currently idle GPU by UUID before importing torch; execute runs sequentially.

```bash
nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv
# Set GPU_UUID to one idle UUID from the inventory.
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?Set GPU_UUID first}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

CP=assignment2/results/reproduce_cartpole/seed33
python -m assignment2.cartpole.train --output "$CP" \
  --seed 33 --episodes 1000 --max-steps 80000
python -m assignment2.cartpole.validate finalize --run "$CP"

MS=assignment2/results/reproduce_mspacman/main_seed41
python -m assignment2.mspacman.probe --output assignment2/results/reproduce_atari_probe
python -m assignment2.mspacman.train --output "$MS" --seed 41 \
  --episodes 4000 --max-steps 2000000 --max-hours 4
python -m assignment2.mspacman.validate finalize --run "$MS"
```

Use fresh output directories. `finalize` reads the run's `metrics/selection.json`,
runs exactly 500 greedy episodes, and creates plots. As an **alternative** to
`finalize`, set `CP_CHECKPOINT` and `MS_CHECKPOINT` to the selected paths in those
JSON files and run evaluation and plotting separately:

```bash
python -m assignment2.cartpole.evaluate --checkpoint "$CP_CHECKPOINT" --output "$CP/evaluation500"
python -m assignment2.cartpole.plots --run "$CP" --checkpoint "$CP_CHECKPOINT" \
  --evaluation "$CP/evaluation500" --device cuda:0
python -m assignment2.mspacman.evaluate --checkpoint "$MS_CHECKPOINT" --output "$MS/evaluation500"
python -m assignment2.mspacman.plotting --run "$MS" --device cuda:0
```

Do not execute both routes into the same evaluation directory. Full plotting
reloads checkpoints; `--preferences-only` for Atari uses saved images and Q-values
without network inference. Use a copy of the evidence directory for that rendering.

## Evaluate the published checkpoints

Install the dependencies and use the single-GPU exports above on **deepdish5**.
These existing evaluation CLIs enforce that hostname and CUDA; they have no
`--device` option. Each command evaluates exactly 500 greedy episodes and requires
a fresh output directory. MsPacman also requires the ROM described above.

```bash
python -m assignment2.cartpole.evaluate \
  --checkpoint assignment2/results/cartpole/seed33/checkpoints/best_ep0942.pt \
  --output assignment2/results/published_cartpole_evaluation500
python -m assignment2.mspacman.evaluate \
  --checkpoint assignment2/results/mspacman/main_seed41/checkpoints/best_ep03114.pt \
  --output assignment2/results/published_mspacman_evaluation500
```

The Python loaders support CPU inference via `load_agent(path, "cpu")` in
`assignment2.cartpole.evaluate` and `assignment2.mspacman.agent`; this does not
change the evaluation CLI restrictions. Published-checkpoint evaluation measures
the selected policies, saved-evidence verification below checks recorded statistics,
and the training commands above reproduce the full learning procedure.

## Validation

Verify published statistics and evidence hashes **without dependencies, checkpoints,
or training**:

```bash
python -m assignment2.verify_public_results
```

With dependencies installed, run the focused CPU correctness suites and new-run
algorithm diagnostics:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest assignment2/tests -q
python -m assignment2.cartpole.validate diagnostics --run "$CP"
python -m assignment2.cartpole.validate robustness --run "$CP"
python -m assignment2.mspacman.validate diagnostics --run "$MS"
python -m assignment2.mspacman.validate short-suite \
  --root assignment2/results/reproduce_atari_validation --steps 100000
```

The [technical guide](assignment2/README.md) gives the full CartPole controlled
suite, ablation flags, expected checks, Atari recovery/rollback tests, and repeated
training-prefix commands. The 18 CartPole and 12 Atari focused tests check specific
software properties; target/replay interventions, seed variation, Bellman traces,
and distribution-shift rollouts provide separate algorithm-level evidence.

## Reproducibility limits

CartPole includes three baseline training seeds and seed-11 target/replay/gamma
ablations. The selected seed 33 is stronger than the other two; it retains two
standard-reset failures and reaches 500 steps in 293/405 altered-state tests.
Per-run tests had been inspected before the final across-seed primary designation,
although the designation uses validation scores.

Only one MsPacman run has the full budget. Its six additional runs use about
100,000 decisions and 30 diagnostic games each. Baseline/slower-decay epsilon
remain near 0.82/0.91; every-update and slower-decay variants select initial
checkpoints. These describe early learning, not fully annealed performance. The
recent-replay paired score-difference interval spans zero. See the original
[CartPole](assignment2/results/cartpole/comparison.csv) and
[MsPacman](assignment2/results/mspacman/validation/comparison.csv) comparisons.

The main Atari run ended at its decision cap after 3,114 games, below the suggested
4,000-6,000-game scale. Ten-seed validation panels are reused, single-frame inputs
hide motion, and cross-device/version bitwise repetition is not guaranteed.
Measured training times on one RTX 5060 Ti were about 180 seconds for CartPole
and 54.48 minutes for Atari; these are hardware-specific.

**Author:** [Yuxuan Liang (`yuxuanliang1`)](https://github.com/yuxuanliang1) |
**Repository:** [MLDS490-Assignment2](https://github.com/yuxuanliang1/MLDS490-Assignment2)
