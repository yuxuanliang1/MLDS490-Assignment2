"""Populate academic report notes only from saved measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def table(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):return 'not available' if np.isnan(value) else f'{value:.4f}'
        return str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(cell(value) for value in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def build(run,validation_root):
    run=Path(run);validation_root=Path(validation_root)
    cfg=read(run/'metrics/config.json');train=read(run/'metrics/training_summary.json')
    audit=read(run/'metrics/gpu_audit_session00.json');env=read(run/'metrics/environment.json')
    replay=read(run/'metrics/replay_memory.json');selection=read(run/'metrics/selection.json')
    evaluation=read(run/'evaluation500/summary.json');diagnostic=read(run/'diagnostics/summary.json')
    edges=read(run/'diagnostics/edge_cases.json');learning=read(run/'metrics/learning_evidence.json')
    episodes=pd.read_csv(run/'logs/episodes.csv');updates=pd.read_csv(run/'logs/updates.csv')
    stages=pd.read_csv(run/'metrics/learning_stages.csv');validation=pd.read_csv(run/'logs/validation.csv')
    prefs=pd.read_csv(run/'metrics/action_preferences.csv');smoke=read(PROJECT/'results/mspacman/smoke/summary.json')
    benchmark=read(PROJECT/'results/mspacman/smoke_batch32/metrics/training_summary.json')
    config_table=table(pd.DataFrame([dict(parameter=key,value=str(value)) for key,value in cfg.items()]))
    stage_table=table(stages[['stage','episode','steps','mean','std_population','median','minimum','maximum']])
    if (validation_root/'comparison.csv').exists():
        comparison=pd.read_csv(validation_root/'comparison.csv')
        comparison_table=table(comparison[['run','episodes','steps','evaluation_mean','evaluation_std','best_validation_mean','final_reference_q','final_epsilon']])
        optimization_table=table(comparison[['run','loss_mean','loss_p95','max_abs_q','elapsed_seconds']])
        paired_path=validation_root/'paired_comparisons.csv'
        paired_table=table(pd.read_csv(paired_path)) if paired_path.exists() else 'Paired uncertainty intervals were not computed.'
        by_name=comparison.set_index('run');base=by_name.loc['baseline_seed41'];target=by_name.loc['target_every_update_seed41']
        recent=by_name.loc['recent_replay_seed41'];slow=by_name.loc['epsilon_slow_seed41']
        replicated=read(validation_root/'replication_summary.json')
        experiment_text=f'''
### B. Target-network stabilization

Hypothesis: removing target lag may destabilize bootstrapped image-value regression.
Changed: hard copy every 5,000 optimizer updates versus every update. Controlled:
seed 41, gamma, architecture, batch size, replay, warmup, optimization frequency,
epsilon schedule, environment, decision budget, selection panel and diagnostic seeds.
Both variants retain separate target modules; every-update copies eliminate useful lag.
The selected short baseline scored {base.evaluation_mean:.4f} and every-update
scored {target.evaluation_mean:.4f} on 30 held-out diagnostic episodes: difference
{target.evaluation_mean-base.evaluation_mean:+.4f}. Largest sampled absolute Q was
{base.max_abs_q:.4f} versus {target.max_abs_q:.4f}; loss P95 was
{base.loss_p95:.6f} versus {target.loss_p95:.6f}. Inspect the curves as well as these
statistics. One seed and a limited budget do not establish universal superiority.
Finite but inflated Q estimates can be scientifically poor without triggering NaN guards.

### C. Replay sensitivity

Hypothesis: highly overlapping, recent-only minibatches reduce diversity and can
destabilize DQN. Changed: uniform random sampling versus the most recent 32
transitions in chronological order; same capacity, warmup, batch size and all
other settings at seed 41. The resulting 30-episode means were
{base.evaluation_mean:.4f} versus {recent.evaluation_mean:.4f}, a change of
{recent.evaluation_mean-base.evaluation_mean:+.4f}. Loss P95 was
{base.loss_p95:.6f} versus {recent.loss_p95:.6f}. These results are specific to the
short, still-exploratory regime; they do not prove replay is universally necessary.

### D. Exploration sensitivity

Hypothesis: slower decay preserves broader action coverage, potentially delaying
short-term exploitation while helping later learning. Changed: linear epsilon
decay length 500,000 versus 1,000,000 decisions, with endpoints unchanged.
All remaining variables are controlled at seed 41. Actual final epsilons were
{base.final_epsilon:.6f} and {slow.final_epsilon:.6f}; 30-episode means were
{base.evaluation_mean:.4f} and {slow.evaluation_mean:.4f}, respectively.
The run ends before either schedule reaches its floor; this is a short-horizon
comparison and cannot determine which schedule wins after full training.

### F. Limited reproducibility

Three independent short training seeds (41, 52, 63) share the same configuration
and evaluation seeds. Mean of the per-run diagnostic means:
{replicated['mean_of_run_means']:.6f}; population SD between run means:
{replicated['population_sd_of_run_means']:.6f}. Within-model SD is reported separately
in the table. These are **short-budget replications**, not three completed
full-budget primary experiments. Full-budget multi-seed reproducibility remains
unmeasured. Overlapping confidence intervals or a noisy best checkpoint are not
evidence of algorithmic equivalence or convergence.

All six short runs and their 30-episode diagnostic evaluations completed. Their
seeds 700000-700029 are distinct from both validation and final 500-test panels.
Models were independently selected on the same ten validation seeds. Compare the
actual step counts: the cap is checked at episode boundaries, so small overshoots
depend on episode length. Selection from a tiny panel is noisy and can overfit.

{comparison_table}

{optimization_table}

Evaluation means/SDs refer to each validation-selected checkpoint. Final-reference
Q is measured on the last network; loss summaries and max-absolute-Q cover the
whole training run. These need not describe the same snapshot. A failed-to-improve
variant may select its initial network; inspect its `selection.json` rather than
assuming the diagnostic score comes from the final weights.

Paired differences (variant minus baseline) use the same 30 diagnostic reset seeds:

{paired_table}

Intervals are exploratory percentile bootstrap intervals conditional on the chosen
models, with 10,000 resamples. They describe reset-seed uncertainty, not uncertainty
across independent training seeds; there is no multiple-comparison correction.
An interval spanning zero does not support ranking the variants at this budget.
'''
    else:
        experiment_text='''
### B-D and F. Controlled ablations and independent seeds

**TODO: no completed short-suite comparison is available yet.** Do not infer that
target/replay/exploration interventions help, or that full-budget reproducibility
has been established. The implemented command runs matched short experiments
sequentially; completed measurements must replace this explicit pending status.
'''
    first=episodes.reward.head(100).mean();last=episodes.reward.tail(100).mean()
    actual_stages=table(validation[['episode','steps','mean','std_population']])
    preferences=prefs[prefs.greedy].groupby('action_name').size().to_dict()
    preference_summary=read(run/'metrics/action_preference_summary.json')
    q_range=(episodes.reference_max_q_mean.min(),episodes.reference_max_q_mean.max())
    reference=read(run/'metrics/reference_protocol.json')
    latest=read(run/'metrics/latest_recovery.json')
    prefix_path=PROJECT/'results/mspacman/same_seed_prefix_reproducibility.json'
    if prefix_path.exists():
        prefix=read(prefix_path)
        prefix_text=f"The repeated seed-41 short baseline matched all {prefix['compared_decisions']:,} recorded decision transitions (including image checksums) in the main run's prefix. At the last common saved checkpoint, episode {prefix['parameter_comparison_episode']}, both online and target parameter maximum discrepancies were zero. This is same-hardware/software repeatability evidence; checksums are not a proof of pixel identity, and it is not a second full-budget run."
    else:prefix_text='Same-seed prefix replay comparison is pending; no exact-repeatability claim is inferred.'
    extended_path=PROJECT/'results/mspacman/smoke_recovery_extended/summary.json'
    if extended_path.exists():
        extended=read(extended_path)
        extended_text=f"The extended GPU recovery experiment exercised {extended['continuous']['optimizer_updates']} optimizer updates and {extended['continuous']['target_updates']} hard target copies (interval 50 for this smoke only). An uninterrupted four-game run, a two-plus-two full resume, and rollback/re-execution from the earlier game-two checkpoint all matched online parameters exactly: resume discrepancy {extended['max_parameter_difference']}, rollback discrepancy {extended['rollback_parameter_difference']}. Archived logs were preserved and the current episode sequence was uniquely 1,2,3,4. This is a controlled software-recovery experiment, not a long-budget Atari replication."
    else:extended_text='Extended GPU rollback/recovery experiment is pending.'
    text=f'''# MLDS 490 Assignment 2, Part (b): MsPacman-v5 Classic DQN

## 1. Objective, course references and scope

The assignment requires an image-based DQN for MsPacman-v5 with gamma=.99,
nine actions, replay and a target network. Performance is assessed through
demonstrated learning/improvement; no invented score threshold is presented as
a course requirement. The three required figures, a non-standard visualization,
edge cases and scientific validation are documented here. Exactly 500 final
evaluation episodes are distinct from periodic model selection.

Authoritative source: `references/mlds_assignment2.pdf`, all three pages, reviewed
again for this part. Week 6 DQN pages 6-15 and 18-20, Lab Week 6 pages 4-10
(including the image-based algorithm on page 10), and Lab Week 5's Bellman/
semi-gradient material were reviewed during this project and remain available.
Classic DQN is the baseline; no pretrained policy, RL framework, Double DQN or
combined submission PDF is substituted. CartPole source, tests, report and results
are preserved and have a separate before/after hash audit. Only the shared README
and ignore rules were extended for Part (b).

## 2. Exact environment and initialization

Requested and actual Gymnasium ID: **`{env['actual_spec_id']}`**. Installed modern
ALE registers `ALE/MsPacman-v5`, so `environment.py` creates an explicit unnamespaced
assignment alias using the identical entry point and kwargs, and rejects a
conflicting registration. A test compares alias and canonical RGB/reward/end-flag
trajectories under the same seed/actions, not merely their names.

The underlying spec is {env['spec_kwargs']}. Thus frameskip=4 occurs only inside
ALE, sticky-action probability=.25, and native maximum episode length=108,000
raw frames. There are no extra frame-skipping wrappers, no-op resets, episodic-life
resets, reward clipping, frame stacking or human rendering. All scores include
every life and the last transition. RGB observations have shape (210,160,3) uint8;
the discrete action names are {', '.join(env['action_meanings'])}.
Action indices are zero-based in this order. Sticky actions can make the executed
action differ from the commanded greedy action; epsilon=0 does not remove this
environment stochasticity.

The ROM was already available in the installed ALE ROM package; no download,
license-acceptance step or ROM redistribution was performed. ROM SHA256:
`{env['rom_sha256']}`. Package-level ROM availability and actual game execution
were verified. No additional claim about third-party redistribution rights is made.

The preflight executed three full random games: 1,359 decisions and 5,427 actual
frames, with scores 280, 120 and 230, all ending at game over with zero lives.
Actual frame increments are measured: final transitions can advance fewer than
four frames, so counters do not simply multiply decisions by four.

## 3. Exact image preprocessing and CNN

The assignment operation is `obs[1:176:2, ::2]`, RGB channel sum, mask pixels whose
sum is 448, integer-divide by 3, subtract 128 and cast to int8. Output is (88,80,1)
int8; intermediate sums use signed int32 to avoid overflow/underflow. The same
pixels are transposed to contiguous CHW (1,88,80) for replay. The model converts
sampled images to float32 **on CUDA** and divides by 128, giving [-1,127/128].
This changes numerical scale, not the assignment's crop or mask. On the actual
initial RGB frame, preflight observed processed values from -128 to 27; the full
legal transform range is -128..127. Black, white and sum-448 edge cases are tested.
The sum mask can remove several colors sharing a channel sum; no semantic object
detector is assumed. Single-frame input omits explicit velocity/motion history.

CNN: Conv2d(1,32,8,stride4), ReLU; Conv2d(32,64,4,stride2), ReLU;
Conv2d(64,64,3,stride1), ReLU; flatten; Linear(2688,512), ReLU; Linear(512,9).
Flattened dimension is computed by a dummy forward, not assumed from another
Atari input size. Actual flattened size {train['flattened_conv_size']}; trainable
parameter count {train['model_parameters']:,}. There is no output softmax.

## 4. Bellman update, replay and target synchronization

For transition `(s,a,r,s',terminated,truncated)`, Classic DQN uses

`y = r + .99 * (1 - terminated) * max_a' Q_target(s',a')`.

The target is calculated under no-grad using the separate target CNN, initialized
from the online weights and permanently excluded from gradients. Huber loss uses
the online Q-value indexed by the actual commanded action. Adam updates online
parameters and clips their gradient norm to 10. Logged gradient norms are
**pre-clipping** norms and may exceed 10. Hard copies occur after every 5,000
optimizer updates, independent of episode count.

Interaction stops on genuine termination **or** truncation. Only genuine game
over masks bootstrapping; losing one life is not artificially terminal. Installed
ALE source separately calls `game_over(with_truncation=False)` and `game_truncated()`.
An external or native time limit is not an absorbing state, so truncated-only
transitions continue to bootstrap. Both flags survive replay and scalar logs.
If both flags are true, genuine termination takes precedence.

Replay preallocates compact int8 state and next-state arrays, action int64,
reward float32, terminated bool and truncated bool. Keeping both images avoids
reconstructing a wrong next image across episode boundaries or ring wraparound.
At capacity {replay['capacity']:,}, exact allocation is {replay['estimated_bytes']:,}
bytes ({replay['estimated_bytes']/2**30:.6f} GiB). This includes both images and
metadata, not the much larger float32 alternative. It is rejected before allocation
if it exceeds 25% of available RAM. Host available RAM at allocation was
{replay['available_host_bytes']/2**30:.3f} GiB. Sampling is uniform without replacement
within each independent minibatch. Sampled int8 batches, not the full replay, move
to CUDA. Ring position and random sampling state are included in recovery snapshots.

## 5. Hyperparameters and defensible budget

{config_table}

One optimizer update occurs every four global **agent decisions** after warmup.
This is the standard update-frequency design choice, separate from native frameskip.
The main cap was explicitly 4,000 complete games, 2,000,000 decisions or four
hours, whichever was reached first at a game boundary. The course's 4,000-6,000
episodes is a suggested scale rather than a guarantee; actual experience and
runtime are reported rather than pretending a step-limited run completed 4,000 games.
No reward shaping/clipping or mixed precision is used. Default epsilon decays
linearly from 1 to .1 over 500,000 decisions, and then stays .1.

## 6. GPU/software verification and measured smoke

Host: `{audit['hostname']}`. GPU: **{audit['device_name']}**, nominal 16 GB,
selected by UUID `{audit['cuda_visible_devices']}` (NVML index 0 in this session),
CUDA logical device 0. The preferred 48 GB PRO 5000 was occupied and was left
untouched. Exactly one physical GPU is visible to each job; expensive jobs run
sequentially. Capability 12.0/Blackwell `sm_120` is supported by the installed build.
Python `{audit['python'].splitlines()[0]}`; PyTorch {audit['versions']['torch']}, CUDA
{audit['torch_cuda']}, Gymnasium {audit['versions']['gymnasium']}, ALE {audit['ale_py']},
NumPy {audit['versions']['numpy']}, Matplotlib {audit['versions']['matplotlib']},
pandas {audit['versions']['pandas']}; pytest 9.2.0 and psutil 7.2.2.
The existing compatible project environment was reused without shared-package changes.
Actual CUDA matrix multiplication returned 3680 as expected; CNN input, Q-values,
targets, loss, gradients and Adam moments were recorded on cuda:0. Environment
emulation and compact replay remain on CPU. No claim of GPU speedup over CPU is made.

Before main training, a four-episode GPU smoke plus two-plus-two resume matched
online parameters exactly: maximum discrepancy {smoke['max_parameter_difference']:.1f}.
The batch-32 smoke completed {benchmark['steps']:,} decisions and
{benchmark['optimizer_updates']:,} updates in {benchmark['elapsed_seconds']:.6f} s,
measuring {benchmark['session_steps_per_second']:.6f} decisions/s and
{benchmark['session_updates_per_second']:.6f} updates/s, including its evaluation/
checkpoint overhead. The initial estimate for the 2-million-decision cap was
{2000000/benchmark['session_steps_per_second']/60:.2f} minutes; it was an estimate,
not a claimed main runtime. Actual throughput changes with exploration and episode length.

## 7. Completed main training and reward progression

Completed **{train['episodes']:,} episodes**, **{train['steps']:,} decisions** and
**{train['raw_frames']:,} measured raw frames**; {train['optimizer_updates']:,}
optimizer updates and {train['target_updates']} target copies. Stop reason:
`{train['stop_reason']}`. Training wall time {train['elapsed_seconds']:.6f} seconds
({train['elapsed_seconds']/60:.3f} minutes), including periodic validation and recovery.
Peak tensor allocation was {train['peak_gpu_allocated_bytes']/2**20:.3f} MiB;
final process RSS was {train['rss_bytes']/2**30:.3f} GiB. GPU driver memory includes
additional context/reserved memory and is not identical to tensor allocation.

Mean score in first 100 training games: **{first:.6f}**; last 100:
**{last:.6f}**; change {last-first:+.6f}. The final trailing-100 mean is the actual
100-game statistic, not a partial window. Final epsilon: {train['epsilon']:.6f}.
Training rewards mix exploration with changing online weights; they are not
the performance distribution of a fixed greedy policy. Read the entire curve,
including regressions, rather than a favorable selected interval.

Mean fixed-reference maximum Q ranged from {q_range[0]:.6f} to {q_range[1]:.6f}.
Largest sampled absolute training Q was {updates.max_abs_q.max():.6f}; Huber loss
mean {updates.loss.mean():.6f}, P95 {updates.loss.quantile(.95):.6f}.
Finite estimates alone do not establish calibrated values or policy convergence.
Discounted raw-score estimates are not undiscounted episode totals; unlike
CartPole, raw rewards are not all +1 and there is no unit-reward Q=100 bound.

## 8. Checkpoint selection and exactly 500 held-out episodes

Selected checkpoint: `{run/'checkpoints'/selection['checkpoint']}` at episode
{selection['episode']}, validation mean **{selection['mean']:.6f}**. The protocol
is fixed: ten greedy seeds 600000-600009 every 100 episodes and at the end;
highest mean wins, earliest wins ties. The final model is separately retained.
The final 500-episode test panel does not select checkpoints or hyperparameters.

Exactly **{evaluation['episodes']}** greedy episodes, epsilon=0, seeds 900000-900499:
mean **{evaluation['mean']:.6f}**, population standard deviation (`ddof=0`)
**{evaluation['std_population']:.6f}**, median {evaluation['median']:.6f}, minimum
{evaluation['minimum']:.6f}, maximum {evaluation['maximum']:.6f}. Mean episode length
{evaluation['mean_decision_steps']:.6f} decisions / {evaluation['mean_raw_frames']:.6f}
raw frames. {evaluation['terminated_episodes']} terminated and
{evaluation['truncated_episodes']} truncated. Evaluation wall time:
{evaluation['elapsed_seconds']:.6f} seconds. All 500 individual scores and end flags
are in `evaluation500/episodes.csv`; no periodic episodes are substituted.

## 9. Required figures and non-standard visualization

Every figure is saved as readable PNG and vector PDF under `{run/'figures'}`:

1. `maximum_q_values`: after each episode, evaluate the same {reference['count']}
   reference images, take max over nine actions, and plot the mean plus/minus
   population SD across images. Reference images were collected with a random
   policy at seed 410000, every 25 decisions, before training; there are
   {reference['unique_images']} unique images. Their fixed SHA256 is
   `{reference['sha256']}`. Never substitute changing replay samples.
2. `training_rewards`: all undiscounted exploratory episode scores and the strict
   trailing-100 mean, with separate greedy validation points. Nothing is clipped.
3. `evaluation_histogram`: precisely the 500 held-out raw scores above, with mean
   and population SD. Sticky actions remain part of the environment.
4. `state_action_preferences` (non-standard): six representative early reference
   RGB frames beside nine predicted Q-values per frame, highlighting the greedy
   commanded action. Underlying values are in `metrics/action_preferences.csv`.
   Full RGB frames are shown for human context; the CNN only sees their specified
   cropped/downsampled/masked single-frame representation, not the full display.
   On the full fixed panel, greedy action counts are {preferences}. The selected
   policy uses {preference_summary['distinct_selected_greedy_actions']} different
   greedy commands and differs from initialization on
   {100*preference_summary['fraction_actions_changed_from_initial']:.3f}% of images.
   Median top-two Q margin is {preference_summary['median_top_two_q_margin']:.6f};
   mean within-image Q range is {preference_summary['mean_within_image_q_range']:.6f}.
   These quantify state dependence and changed preferences without interpreting
   a Q margin as a probability. A small number of dominant commands would qualify
   claims about visual scene understanding. These visual
   preferences are not ground-truth optimal actions; the panel comes from early
   random play, may omit later levels, and a single frame hides motion. Similar
   preferences across different frames may indicate a limited policy rather than
   successful scene understanding. Actual returns provide complementary evidence.
5. `optimization_diagnostics`: loss, global-step epsilon and pre-clipping gradient
   norm. `learning_stages` compares common-seed validation at distinct stages.

## 10. Scientific validation design and measured findings

### A. Evidence of learning

Hypothesis: DQN updates produce a policy with improved greedy raw score relative
to random initialization. Controlled: exact environment, preprocessing, architecture,
validation reset seeds and greedy evaluation. Changed: training exposure/weights.

{stage_table}

Initial mean {learning['initial_validation_mean']:.6f}; selected mean
{learning['selected_validation_mean']:.6f}; paired improvement
{learning['paired_validation_improvement']:+.6f}. An exploratory paired-bootstrap
95% interval over the reused ten-seed panel is
{learning['paired_bootstrap_95_percent_interval']}. This interval is selection-biased
and must not be presented as an unbiased confirmatory improvement interval.
Uniform-random control on the same reset seeds averaged
{learning['random_control']['mean']:.6f}, population SD
{learning['random_control']['std_population']:.6f}. Initial greedy evaluation can
be nearly deterministic despite different reset seeds if a constant command
renders sticky-action randomness ineffectual. Training trajectory, validation
trajectory and the untouched 500-test distribution are distinct evidence sources.

Full periodic validation history (including decreases):

{actual_stages}

{experiment_text}

### E. Independent Bellman arithmetic and gameplay trace

Hypothesis: targets use raw reward, gamma=.99, maximum target-network next Q,
and genuine termination only, while action selection follows online argmax.
The independently evaluated float64 scalar formula was compared against the
implementation on {diagnostic['actual_transition_count']} actual replay transitions,
including {diagnostic['terminal_samples']} genuine terminals and
{diagnostic['truncated_samples']} truncated transitions. Maximum absolute discrepancy:
**{diagnostic['max_bellman_absolute_error']:.9g}** (atol 1e-3, rtol 2e-6).
This verifies arithmetic/masking using saved target Q-values, not an independent
CNN implementation or historical targets from when those transitions were collected.
The actual image transitions and per-transition reference CSV are saved.

At reset seed 314159, {diagnostic['first_ten_steps']} first steps and
{diagnostic['later_window_steps']} later-window steps were traced, including all
nine Q-values, chosen action, raw reward, flags, lives and target. Every trace
row links to exact state/next-state image arrays. Argmax matches:
{diagnostic['greedy_argmax_matches']}. Initial startup frames may repeat or yield
no reward; the later window complements this uninformative but legitimate edge.
CPU/CUDA reload maximum Q difference was
{diagnostic['cpu_cuda_q_max_absolute_difference']:.9g}, and action agreement
{100*diagnostic['cpu_cuda_greedy_agreement']:.3f}%. Cross-device bitwise equality is
not assumed. Analytic unit cases additionally distinguish genuine termination
from native and wrapper time limits, even if the main run observed no truncations.

## 11. Atari-specific edge cases and correctness evidence

The focused suite has **12 passing tests**. It tests exact pixel arithmetic,
signed range and CHW layout; all nine valid actions; invalid actions/shapes;
fixed-buffer underflow/wraparound, separate next images and flags; memory-allocation
guard; known chosen-action Huber values; detached targets, freezing and exact hard
synchronization; epsilon extremes/global-step schedule; model/full checkpoint
reload; Python/NumPy/PyTorch/replay/behavior RNG restoration; rollback of uncommitted
logs; canonical/assignment-alias equivalence; and native ALE versus wrapper truncation.
The native truncation edge test uses a separate 12-frame cap, and the wrapper test
uses nine decisions; neither changes the primary environment's 108,000-frame cap.

Measured main-game edge cases: {edges['game_over_transitions']} genuine game-over
transitions, {edges['truncated_transitions']} truncated transitions, and
{edges['life_loss_nonterminal_transitions']} life losses that correctly continued
the same game. Zero-reward fraction {edges['zero_reward_transition_fraction']:.6f};
largest zero-reward streak {edges['max_zero_reward_streak']} decisions. Such streaks
are legitimate, not reasons to fabricate a reward or end a game early. Raw reward
range [{edges['minimum_transition_reward']:.0f},{edges['maximum_transition_reward']:.0f}],
with counts {edges['actual_reward_values']}. These rewards remain unclipped.
Unchanged processed-image fraction by checksum:
{edges['unchanged_processed_frame_fraction']:.6f}; checksums can in principle collide
and are a diagnostic, not a proof of exact pixel identity. Black/white input probes
produced finite Q-values. Commanded-action counts: {edges['action_counts']}.

Nonfinite Q/loss/targets or gradient norms stop the run with diagnostics, rather
than skipping a poisoned update. Invalid actions and image shapes fail clearly.
The target never receives gradients. Successful software tests establish specific
properties, not a guarantee that the overall learning algorithm converges.

## 12. Recovery, persistence and failure accounting

Model-only snapshots and previous best checkpoints are immutable. Full recovery
is at complete game boundaries and includes online/target weights, optimizer,
update counters, global epsilon step, replay arrays/ring position/sampling RNG,
behavior RNG and Python/NumPy/PyTorch/CUDA RNG. Fixed-reference hashes and selection
metadata are restored. The next environment reset seed is deterministic from the
next episode index, so an in-flight ALE clone is unnecessary at a boundary.
Full snapshot: `{latest['checkpoint']}` plus `{latest['replay_file']}`;
compressed replay size {latest['replay_bytes']:,} bytes. Full replay snapshots are
not promised at every lightweight model checkpoint.
Keep the run's reference-image files, metadata and retained best models alongside
the recovery pair. The early primary schema stores the best validation score and
reconstructs its selection from retained best checkpoints; later snapshots also
store the selection pointer directly. Moving only the pair loses those external
reference/selection artifacts, although it contains optimizer and replay state.

If a crash leaves log rows beyond the restored checkpoint, those files are archived
before rollback; resumed checkpoints receive session suffixes, preserving older
successes. A model-only restore is explicitly rejected as a full resume. SIGINT/
SIGTERM request saving at the next game end; abrupt termination can lose the
in-flight game and work after the most recent full snapshot. GPU smoke measured
exact continuation for its configuration; cross-version/device exactness is not
guaranteed. No shared environment or other user's process was modified/stopped.

{prefix_text}

{extended_text}

Initial preflight logging exposed NumPy scalar JSON serialization issues and the
first detached-launch attempt exposed argument parsing. They were fixed before
main training and are retained in logs; neither is counted as an algorithm experiment.
The first preflight attempts did execute games but did not have successful process
exit status. Successful preflight, smoke, tests and job return codes are the evidence
used here. A persistent PID alone is never treated as proof of completion.

## 13. Strengths, limitations and interpretation

The measured first-versus-last training means and common-seed greedy comparisons
quantify improvement (or regression) directly; larger Q-values or a lower Huber
loss are not proxies for better play. The entire history is retained. The final
policy may remain suboptimal due to partial observability from one frame, lossy
assignment preprocessing, limited experience/replay, sparse/irregular rewards,
sticky actions, bootstrap approximation and a small reused selection panel.
Only one primary training seed is full-budget; short replications cannot remove
that limitation. A bounded run need not complete the suggested 4,000-6,000 games.
No universal convergence, calibrated-Q or optimal-control claim follows from these
results. Controlled interventions and their possibly negative/inconclusive results
are reported separately from implementation bugs.

## 14. Additional three months

Month 1: preregister larger matched-frame/decision budgets and rotate validation
panels; train 10-20 full independent seeds; reserve an untouched final test; report
sample efficiency, run-to-run success distributions and paired uncertainty. Profile
emulation, transfers and CNN kernels; compare single-frame and four-frame inputs
while controlling replay memory and experience, without silently changing v5 dynamics.

Month 2: sweep target lag, replay capacity, exploration decay, update ratio and
learning rate with controlled compute. Add explicitly named Double DQN, dueling
heads, prioritized replay and n-step returns as separate ablations. Compare their
value estimates with empirical discounted returns, not just their final raw scores.
Investigate reward scaling versus clipping while continuing to report raw scores.

Month 3: inspect later-level, frightened-ghost, life-loss and long no-reward behavior;
build stratified visual action-preference and saliency diagnostics; test observation
corruption, different sticky-action seeds and exact recovery across interruptions.
Evaluate richer representations, frame stacking and noisy-network exploration.
Replicate with independent code/hardware, publish raw/negative results, and assemble
the combined report only after the author supplies public GitHub metadata.

## 15. Exact commands and artifacts

Execute from the workspace root on deepdish5. Reinspect GPUs and select one idle
UUID before importing torch. Use fresh run/job labels when repeating experiments.

```bash
export CUDA_VISIBLE_DEVICES=GPU-cef65388-88b0-f39a-6bac-c5316eb78540
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest assignment2/tests/test_mspacman.py -q
.venv/bin/python -m assignment2.mspacman.smoke --root assignment2/results/mspacman/smoke
.venv/bin/python -m assignment2.mspacman.train --output {run} --seed 41 --episodes 4000 --max-steps 2000000 --max-hours 4
.venv/bin/python -m assignment2.mspacman.validate finalize --run {run}
.venv/bin/python -m assignment2.mspacman.validate short-suite --root {validation_root} --steps 100000
.venv/bin/python -m assignment2.mspacman.validate same-seed-prefix --main {run} --short {validation_root}/baseline_seed41
.venv/bin/python -m assignment2.mspacman.smoke --root assignment2/results/mspacman/smoke_recovery_extended --rollback-test
.venv/bin/python -m assignment2.mspacman.report --run {run}
.venv/bin/python -m assignment2.mspacman.audit --run {run}
```

Detached execution, individual evaluation/plot commands, and full resume commands
are in the main README. `jobs/*.json` retains PID, actual command, GPU visibility
and exit status; command ledgers retain durations/return codes. Figure input CSVs,
fixed reference `.npy`, 500 episode CSV, image transition diagnostics, immutable
models and full recovery pairs are retained. Large weights/replay/raw transition
logs are ignored by Git; do not publish ROMs or course PDFs. CartPole preservation
is measured with SHA256 before/after, separately from the newly extended README.
No final combined PDF was assembled in this phase.
'''
    (PROJECT/'REPORT_NOTES_MSPACMAN.md').write_text(text,encoding='utf-8')
    print(PROJECT/'REPORT_NOTES_MSPACMAN.md',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='assignment2/results/mspacman/main_seed41')
    p.add_argument('--validation-root',default='assignment2/results/mspacman/validation')
    a=p.parse_args();build(a.run,a.validation_root)


if __name__=='__main__':main()
