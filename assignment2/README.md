# Technical reproduction guide

This is the detailed companion to the [repository README](../README.md) and the
[final report](../MLDS_490_Assignment_2.pdf). Execute commands from the repository
root with Python 3.12 and the project environment activated. The package remains
`assignment2`; do not change its import paths. Both parts are implemented.

## Environment and GPU selection

The recorded environment was Python 3.12.6, PyTorch 2.14.0/CUDA 13.0,
Gymnasium 1.4.0, ALE 0.10.2, NumPy 2.5.3, Matplotlib 3.11.2, pandas 3.0.5,
pytest 9.2.0, and psutil 7.2.2. Install from the root requirements file into a
project environment. Shared installations need not be modified.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hostname
pwd
nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.used,utilization.gpu --format=csv
# Assign GPU_UUID to exactly one currently idle device from the inventory.
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
python -m assignment2.cartpole.runtime --output gpu_audit.json
```

The audit and substantive experiment CLIs require deepdish5 and one usable CUDA
device, including a real known-answer tensor computation. Select by UUID because
CUDA/NVML ordinal ordering differed during the recorded session. The measured
GPU was an RTX 5060 Ti 16 GB, capability 12.0; the build included `sm_120`.
Environment stepping/replay storage are CPU work; network optimization is CUDA.
Training does not fall back to CPU. Do not stop another user's GPU process.

For ALE, the measured package already provided the ROM. Verify it with:

```bash
python -c "from ale_py import roms; print(roms.get_rom_path('ms_pacman').name)"
```

No ROM binary is included in this repository. Use the official ALE installation
instructions for the installed release if the check fails; do not substitute a
different game version or silently change emulator settings.

## Lightweight correctness checks

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest assignment2/tests/test_cartpole_dqn.py -q
CUDA_VISIBLE_DEVICES=-1 python -m pytest assignment2/tests/test_mspacman.py -q
python -m assignment2.verify_public_results
```

The focused suites test known Bellman/Huber values, terminal versus truncated
masking, final-transition reward counting, replay underflow/wraparound, epsilon
extremes, hard/soft target updates, target freezing, nonfinite guards, image
arithmetic, alias equivalence, native/wrapper limits, and checkpoint/recovery
behavior. These tests are separate from the scientific comparisons in the report.
The public verifier needs only the Python standard library and saved metrics.

## CartPole: one run

Always use a new output directory. The committed evidence uses seed 33 as the
selected primary, while seed 11 is the matched control for ablations.

```bash
CP=assignment2/results/new_cartpole
python -m assignment2.cartpole.train --output "$CP/seed33" \
  --seed 33 --episodes 1000 --max-steps 80000
python -m assignment2.cartpole.validate finalize --run "$CP/seed33"
```

`finalize` selects via the run's selection JSON, runs exactly 500 greedy test
resets, makes plots, and evaluates a separate 100-reset final-snapshot diagnostic.
Alternatively, run evaluation/plots individually with `CHECKPOINT` set to the
path recorded in `metrics/selection.json`:

```bash
python -m assignment2.cartpole.evaluate --checkpoint "$CHECKPOINT" \
  --output "$CP/seed33/evaluation500"
python -m assignment2.cartpole.plots --run "$CP/seed33" --checkpoint "$CHECKPOINT" \
  --evaluation "$CP/seed33/evaluation500" --device cuda:0
python -m assignment2.cartpole.validate diagnostics --run "$CP/seed33"
python -m assignment2.cartpole.validate robustness --run "$CP/seed33"
```

Do not execute `finalize` and individual evaluation into the same output.
The actual selected model was `best_ep0942.pt` (not published); reruns can select
a different episode. Training limits are explicit because the current generic
Config step default is larger. Q statistics evaluate each episode's initial
observation using that episode's final online weights. A strict trailing mean
first appears at episode 100. The action map fixes cart position/velocity at zero.

## CartPole: full controlled suite

This is an alternative to the single-run route above. Start a fresh root;
otherwise the suite's seed-33 output would collide with the preceding example.

```bash
CP=assignment2/results/new_cartpole_suite
python -m assignment2.cartpole.train --output "$CP/main_seed11" \
  --seed 11 --episodes 1000 --max-steps 80000
python -m assignment2.cartpole.validate finalize --run "$CP/main_seed11"
python -m assignment2.cartpole.validate suite --root "$CP" --max-steps 80000
python -m assignment2.cartpole.validate primary --root "$CP"
python -m assignment2.cartpole.validate compare --root "$CP"
```

The suite sequentially trains seeds 22/33, soft targets, every-update targets,
recent-only replay, and gamma .90/.99, evaluating each selected model on 500
resets. All baseline settings are fixed except the declared intervention.
Individual ablation overrides for `cartpole.train` are:

| Experiment | CLI overrides (in addition to seed 11 and common budgets) |
|---|---|
| Soft target | `--target-mode soft --target-tau 0.999` |
| Every-update hard copy | `--target-interval 1` |
| Correlated replay | `--replay-mode recent` |
| Discount sensitivity | `--gamma 0.90` or `--gamma 0.99` |

Soft tau .999 weights the **old** target. Recent replay uses the newest 128
chronological records with overlapping minibatches; capacity remains 50,000.
All runs have 1,000-episode/80,000-step caps, checked at episode boundaries.
The every-update and gamma-.99 runs hit the episode ceiling below 80,000 steps.
Validation resets are 50000-50009; test resets are 100000-100499. The stress grid
uses 70000-70004 with modified initial angle/angular velocity and unchanged dynamics.

## MsPacman: verification, smoke, training, evaluation

```bash
python -m assignment2.mspacman.probe --output assignment2/results/new_atari_probe
python -m assignment2.mspacman.smoke --root assignment2/results/new_atari_smoke
MS=assignment2/results/new_mspacman/main_seed41
python -m assignment2.mspacman.train --output "$MS" --seed 41 \
  --episodes 4000 --max-steps 2000000 --max-hours 4
python -m assignment2.mspacman.validate finalize --run "$MS"
```

The literal `MsPacman-v5` is registered as a checked alias of
`ALE/MsPacman-v5`: native frameskip 4, sticky probability .25, nine actions,
108,000 raw-frame limit, no frame stack, no additional repeat wrapper, no reward
clipping or episodic-life resets. Preprocessing is crop/downsample, signed RGB
sum, sum-448 suppression, integer division by 3, subtraction of 128, int8 CHW;
the CNN casts and divides by 128. Life loss continues the same game.

The recorded main run completed 3114 games, not 4000. Decisions and raw frames
are separately counted. Validation uses 600000-600009; exactly 500 final tests
use 900000-900499. The maximum-Q curve averages max-action Q over a fixed panel
of 32 early random-play images; its band is SD across images, not across runs.

Individual evaluation/plot commands, as an alternative to `finalize`:

```bash
python -m assignment2.mspacman.evaluate --checkpoint "$CHECKPOINT" --output "$MS/evaluation500"
python -m assignment2.mspacman.plotting --run "$MS" --device cuda:0
python -m assignment2.mspacman.validate learning-stages --run "$MS"
python -m assignment2.mspacman.validate diagnostics --run "$MS"
```

Set `CHECKPOINT` from selection JSON; the original choice was `best_ep03114.pt`.
Full plotting reloads a model; it is not checkpoint-free. For the published
reference RGB images and previously measured Q values only:

```bash
python -m assignment2.mspacman.plotting \
  --run assignment2/results/mspacman/main_seed41 --preferences-only
```

This CPU Matplotlib route performs no new network inference but writes the
preference figure. Use a copy of the evidence directory when preserving its files.

## MsPacman: short scientific suite and recovery

```bash
python -m assignment2.mspacman.validate short-suite \
  --root assignment2/results/new_atari_validation --steps 100000
python -m assignment2.mspacman.smoke \
  --root assignment2/results/new_recovery_smoke --rollback-test
```

Six matched-cap runs cover seeds 41/52/63, target copy every update, newest-32
chronological replay, and epsilon decay over 1,000,000 rather than 500,000
steps. Each selected model receives 30 diagnostic resets 700000-700029.
These are short, still-exploratory comparisons, not full-budget replication.
Paired bootstrap intervals quantify reset variation conditional on selected
models; they are not training-seed intervals. Correlated-replay differences
are inconclusive at this budget.

After regenerating both full and short transition logs and common saved stages:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m assignment2.mspacman.validate same-seed-prefix \
  --main "$MS" --short assignment2/results/new_atari_validation/baseline_seed41
```

The original match covers 100031 recorded transitions and episode-100 saved
online/target parameters. Those large logs/checkpoints are omitted publicly.

For a new run's full recovery, set `RECOVERY_CHECKPOINT` to the path in
`metrics/latest_recovery.json` and retain its replay sidecar, reference files,
metadata and previous best models:

```bash
python -m assignment2.mspacman.train --output "$MS" --resume "$RECOVERY_CHECKPOINT" \
  --episodes 6000 --max-steps 3000000 --max-hours 4
```

Repeat any nondefault non-budget settings. A model-only file is not a full resume.
Recovery restores optimizer/replay/global epsilon step/RNG at a game boundary;
rows beyond that point are archived before rollback, and resumed model names
have session suffixes. Abrupt termination can lose the current game and work
since the last recovery point. Exact reproduction is not promised across hardware.

## Detached execution and historical utilities

A portable job launcher retains its own PID, output log and exit code, refusing
active conflicting drivers and duplicate job labels:

```bash
python assignment2/launch_mspacman.py new_training -- \
  .venv/bin/python -m assignment2.mspacman.train --output "$MS" \
  --seed 41 --episodes 4000 --max-steps 2000000 --max-hours 4
```

The CartPole shell wrappers and `launch.py` preserve the original experiment
workflow. They use the caller's selected CUDA visibility and recorded output
names, so they are for a fresh artifact tree, not a checkout containing those
results. `build_report.py`, `verify_results.py`, and the Atari audit/report
utilities are historical full-artifact tools: they need omitted checkpoints,
update/transition logs, or job records and may regenerate ignored development
notes. They do not regenerate or modify the final root PDF. For the public
subset, use `verify_public_results` instead of a full-artifact audit.

## Data availability and model limitations

See [PUBLIC_RESULTS.md](PUBLIC_RESULTS.md). Selected CSV/JSON evidence and PNG
figures are committed; full per-step logs, raw optimizer logs, all checkpoints,
replay recovery archives, job metadata, course PDFs and editable LaTeX remain
local. Fixed Atari reference arrays are included because they are a small,
non-training diagnostic panel. They are not a substitute for model weights.

Inference with the exact selected policies requires obtaining their unpublished
checkpoints. Training anew can give a different policy. Seed-33 selection used
validation, but per-run CartPole tests had been inspected before final across-seed
primary designation; the report discloses this limitation. There is no universal
convergence, robustness, or cross-device determinism claim.
