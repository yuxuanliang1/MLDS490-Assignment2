# Public evidence and data availability

The committed subset preserves original measurement files. It supports review of
both final evaluations, training curves, matched comparisons, and selected
diagnostics without rerunning training. `PUBLICATION_MANIFEST.json` at the root
lists SHA256 hashes and sizes for the current final PDF and selected original
measurement files. New runs are ignored until explicitly selected for publication.

## Included experiments

| Game | Public runs and evaluation protocol |
|---|---|
| CartPole | `main_seed11`, `seed22`, `seed33`, `soft_target_seed11`, `target_every_update_seed11`, `recent_replay_seed11`, `gamma090_seed11`, `gamma099_seed11`; 500 test episodes per selected model |
| MsPacman primary | `main_seed41`; 500 final test episodes and the ten-reset validation panels |
| MsPacman short suite | `validation/baseline_seed41`, `seed52`, `seed63`, `target_every_update_seed41`, `recent_replay_seed41`, `epsilon_slow_seed41`; 30 diagnostic episodes per selected model, about 100,000 training decisions each |

The main CartPole result is seed 33. Its ablations use seed 11 as the matched
control; comparisons should not substitute seed 33 for that control. The short
Atari seeds are not full-budget replications. All standard deviations in result
tables are within-model population SD unless labeled as variation between run means.

For these runs, public data include saved configurations, selection metadata,
training summaries, episode-level training/validation CSVs, individual evaluation
returns and summaries, and root comparison tables. CartPole's separate 100-reset
final-snapshot diagnostics remain clearly distinguished from its selected-model test.
The selected primary runs include PNG figures, numerical policy data, trajectory
and Bellman-reference CSVs, and edge/stress diagnostics.

Atari's 32-image fixed reference panel (processed and RGB arrays) is included for
the non-standard plot and fixed-reference protocol. It is a small diagnostic panel,
not the replay memory. Small sampled transition/trajectory image archives link
the saved Atari diagnostic rows to actual inputs. Load NumPy arrays with
`allow_pickle=False`. The eight principal PNG figures are:

- `results/cartpole/seed33/figures/initial_state_max_q.png`
- `results/cartpole/seed33/figures/training_rewards.png`
- `results/cartpole/seed33/figures/evaluation_histogram.png`
- `results/cartpole/seed33/figures/policy_action_map.png`
- `results/mspacman/main_seed41/figures/maximum_q_values.png`
- `results/mspacman/main_seed41/figures/training_rewards.png`
- `results/mspacman/main_seed41/figures/evaluation_histogram.png`
- `results/mspacman/main_seed41/figures/state_action_preferences.png`

Same-seed-prefix and a compact derived recovery-smoke summary are also included.
The recovery summary omits private interpreter paths and command records, retains
the measured discrepancies/counters, and records its original source SHA256.
Those summaries are historical evidence; the
public subset cannot independently replay the full state restoration without
the omitted recovery artifacts.

## Deliberately omitted

All model weights, replay recovery snapshots, full decision-level transition logs,
full optimizer-update logs, transient launch/test logs, PID/status files, driver
commands containing machine paths, raw GPU inventory dumps, transfer archives,
local development manifests/notes, course handouts, and editable report sources
remain local. Only the supplied final root report is distributed as PDF; redundant
figure PDFs are omitted in favor of PNGs. No credentials, ROM binaries, or virtual
environments are included. Excluded artifacts were not deleted.

**Exact selected-policy inference is not available from this checkout alone.**
Obtain the original checkpoints separately, or retrain using the recorded
configuration; retraining may produce different policies. The historical filenames
are `seed33/checkpoints/best_ep0942.pt` for CartPole and
`main_seed41/checkpoints/best_ep03114.pt` for MsPacman. Full Atari resume additionally
needs the full recovery checkpoint/replay pair, fixed references, metadata, retained
best models, and matching software. No external model-download URL is asserted.

## Verification

```bash
python -m assignment2.verify_public_results
```

This read-only standard-library command checks evidence hashes, both main 500-row
evaluation panels, all eight CartPole and six Atari short comparison evaluations,
training counts, Atari first/final-100 means, CartPole stress outcomes, and saved
Bellman-error CSVs. It does not import torch, run environments, recompute network
outputs, validate every historical optimizer update, or reconstruct old TD targets.
The full-artifact audit tools require the omitted local archive and are not the
appropriate verifier for a clean public clone.
