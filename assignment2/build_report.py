"""Build measured report notes from experiment artifacts; no invented results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results/cartpole'


def read(path):
    return json.loads(Path(path).read_text())


def markdown_table(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return 'not reached' if np.isnan(value) else f'{value:.4f}'
        return str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def report():
    selection=read(RESULTS/'primary_selection.json')
    run=RESULTS/selection['run']
    config=read(run/'metrics/config.json')
    training=read(run/'metrics/training_summary.json')
    evaluation=read(run/'evaluation500/summary.json')
    audit=read(run/'metrics/gpu_audit.json')
    diagnostic=read(run/'diagnostics/summary.json')
    stress=read(run/'robustness/summary.json')
    across=read(RESULTS/'across_seeds.json')
    comparison=pd.read_csv(RESULTS/'comparison.csv')
    episodes=pd.read_csv(run/'logs/episodes.csv')
    grid=pd.read_csv(run/'metrics/policy_grid.csv')
    stress_grid=pd.read_csv(run/'robustness/grid.csv')
    final100=read(run/'metrics/final_snapshot_diagnostic100.json')
    baseline=comparison.set_index('run').loc['main_seed11']
    config_table=markdown_table(pd.DataFrame([{'Parameter':k,'Value':v} for k,v in config.items()]))
    performance=markdown_table(comparison[['run','episodes','env_steps','evaluation_mean','evaluation_std','success_fraction','final_snapshot_mean']])
    stability=markdown_table(comparison[['run','first_periodic_mean475_episode','loss_mean','loss_p95','max_abs_q','final_initial_q','periodic_mean_std']])
    by_name=comparison.set_index('run')
    soft=by_name.loc['soft_target_seed11']; every=by_name.loc['target_every_update_seed11']
    recent=by_name.loc['recent_replay_seed11']; g90=by_name.loc['gamma090_seed11']; g99=by_name.loc['gamma099_seed11']
    commands=[]
    for candidate in ['main_seed11','seed22','seed33','soft_target_seed11','target_every_update_seed11','recent_replay_seed11','gamma090_seed11','gamma099_seed11']:
        cfg=read(RESULTS/candidate/'metrics/config.json')
        changed=[]
        for key in ['gamma','learning_rate','target_mode','target_tau','target_interval','replay_mode']:
            if cfg[key]!=read(RESULTS/'main_seed11/metrics/config.json')[key]:
                changed.extend(['--'+key.replace('_','-'),str(cfg[key])])
        commands.append('.venv/bin/python -m assignment2.cartpole.train --output assignment2/results/cartpole/'+candidate+
            f" --seed {cfg['seed']} --episodes {cfg['episodes']} --max-steps {cfg['max_steps']} "+' '.join(changed))
    action_examples=[]
    for theta,omega in [(-.05,0),(.05,0),(0,-.5),(0,.5),(-.1,-1),(.1,1)]:
        nearest=grid.iloc[((grid.theta_rad-theta)**2+(grid.omega_rad_s-omega)**2).argmin()]
        action_examples.append(dict(theta_rad=nearest.theta_rad,omega_rad_s=nearest.omega_rad_s,
                                    q_left=nearest.q_left,q_right=nearest.q_right,action=int(nearest.action)))
    probes=markdown_table(pd.DataFrame(action_examples))
    # Physical intuition is a local heuristic, not a ground-truth policy label.
    center=grid[(grid.theta_rad.abs()<=.05)&(grid.omega_rad_s.abs()<=.5)]
    heuristic=(center.theta_rad+.2*center.omega_rad_s)>0
    agreement=float((center.action.to_numpy()==heuristic.to_numpy()).mean())
    low=stress_grid.loc[stress_grid['mean'].idxmin()]
    high=stress_grid.loc[stress_grid['mean'].idxmax()]
    n=500;p=evaluation['success_fraction'];z=1.96
    denominator=1+z*z/n
    center_ci=(p+z*z/(2*n))/denominator
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    header=f'''# MLDS 490 Assignment 2, Part (a): CartPole-v1 DQN

## Submission metadata and course sources

GitHub username: **TODO: author must supply**. Public repository name and URL:
**TODO: author must create/publish and supply**. This work does not publish a repository.
These notes and vector figures are ready to incorporate into the required PDF report;
a final authored submission PDF and Canvas upload remain author tasks.

Read `references/mlds_assignment2.pdf` (all three pages), `Week 6 DQN.pdf`
(pages 6-15 and 18-20), `MLDS_Lab_Week_6_1.pdf` (pages 4-10), and
`MLDS_Lab_Week_5 (3).pdf` (all pages). PDFs appeared after initial workspace inspection;
they were read before substantive training. The image containing Lab 6 page 10's
algorithm was extracted and visually inspected. The assignment is authoritative.
It allocates 30 points to edge cases and validation and requires a public GitHub link.
Only CartPole-v1 is implemented; MsPacman remains outside scope.

## Objective and algorithm

The objective is 500 undiscounted unit rewards in the native CartPole-v1 environment.
The state is float32 `(x, x_dot, theta, theta_dot)` without normalization. Actions
0 and 1 push left and right. The explicit neural network is
`Linear(4,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,2)`.
Its outputs are action values; there is no softmax. The online policy is greedy
for evaluation and epsilon-greedy during interaction.

For uniformly sampled replay transitions, the detached semi-gradient target is

`y = r + gamma * (1 - terminated) * max_a Q_target(next_state, a)`.

Huber loss compares the online Q-value indexed by the observed action to this
target. Adam updates online parameters, with gradient norm clipping at 10. The
logged norm is measured before clipping and can therefore exceed 10. The
separate target starts with identical weights, has gradients disabled, and is
copied every 500 optimizer updates in the baseline. The replay buffer preallocates
fixed NumPy arrays, overwrites the oldest records at capacity, and samples uniformly
without replacement within a minibatch. Each minibatch is newly sampled.

Gymnasium's `terminated` denotes actual failure; `truncated` denotes the 500-step
time limit. Interaction ends on either flag, and the final reward is always counted.
Only genuine termination masks bootstrapping. Treating a time limit as absorbing
would set a valid continuation target to immediate reward alone. If both flags
are true, genuine termination masks the continuation. Both flags are preserved
in transition, replay, episode, and evaluation data. No step occurs after an end.

## Primary model and hyperparameters

The primary model is **{selection['run']}**, selected from the prespecified baseline
seeds 11, 22, and 33 by the highest **validation** mean, {selection['validation_mean']:.3f}.
The selection function does not read held-out evaluation rewards. Within each run,
the highest validation mean selects the checkpoint; ties retain the earliest.
The fixed validation panel uses reset seeds 50000-50009. Final test seeds are
100000-100499. Selecting among seeds adds validation selection bias; all seed results
are therefore reported rather than treating the chosen seed as typical.

Selected checkpoint: `{selection['checkpoint']}`.
The selected episode is {evaluation['checkpoint_episode']}; the run completed
{training['episodes']} episodes, {training['env_steps']:,} steps,
{training['optimizer_updates']:,} optimizer updates and {training['target_updates']} target copies.
The cap was initially 1,000 episodes, with an additional modest 80,000-step budget;
the cap is checked at episode boundaries and may overshoot by up to 499 steps.
Stopping reason: `{training['stop_reason']}`. Early convergence stopping was disabled.

{config_table}

The default configuration was not silently tuned after viewing test rewards.
The three baseline seeds share all settings except the training/reset seed.
Gamma changes occur only in explicitly named sensitivity experiments.

## Hardware, execution, and reproducibility

Training and all substantive experiments ran sequentially on `{audit['hostname']}`,
using one **{audit['device_name']}**, nominally 16 GB (16,311 MiB visible in NVML).
Physical identity: `{audit['cuda_visible_devices']}`, physical NVML index 0 in this
session, PCI bus 00000000:01:00.0. CUDA logical device: 0; capability 12.0.
The other RTX 5060 Ti and RTX PRO 5000 were idle when inspected. UUID selection
avoids an observed NVML/CUDA enumeration mismatch. No other user's process was stopped.

Python: `{audit['python'].splitlines()[0]}`. PyTorch: {audit['versions']['torch']};
compiled CUDA: {audit['torch_cuda']}; driver 615.71.09. The architecture list includes
`sm_120`. Gymnasium {audit['versions']['gymnasium']}, NumPy {audit['versions']['numpy']},
Matplotlib {audit['versions']['matplotlib']}, pandas {audit['versions']['pandas']}; pytest 9.2.0.
The project environment reuses compatible packages read-only and adds pytest locally.
Shared installations were not modified. The CUDA audit's known 4x4 matrix-product
sum was {audit['cuda_matmul_sum']:.1f} (expected 3680).

Model and target parameters and gradients were on `cuda:0`; Adam moments were on
CUDA, while its scalar step counter can remain on CPU. The training code places
minibatch inputs, forward pass, target, loss, backward pass and optimization on
the selected CUDA device. Later per-run summaries additionally record loss/target
devices explicitly. CPU stepping is expected for Gymnasium. Primary training took
{training['elapsed_seconds']:.3f} seconds, excluding subsequent evaluation and plots.
Total substantive training time across completed runs was
{comparison.runtime_seconds.sum():.3f} seconds ({comparison.runtime_seconds.sum()/60:.2f} minutes).
The small network need not be faster on GPU than CPU; no speedup claim is made.

Seeds cover Python, NumPy, PyTorch, CUDA, replay sampling, action spaces and each
training/evaluation reset. Deterministic algorithms are requested and cuBLAS workspace
configuration is set before import. Separate training and evaluation environments
prevent evaluation from advancing the training state or RNG. Different versions,
hardware and floating-point kernels can still change trajectories. Checkpoints
support CPU/GPU inference reload; full optimizer/replay/RNG resume is not implemented.
Persistent drivers retain PID, output logs, subprocess return codes and exit files.

## Exactly 500 held-out greedy episodes

The primary policy used epsilon=0 and 500 distinct reset seeds, 100000-100499.
Results: mean **{evaluation['mean']:.6f}**, population standard deviation (`ddof=0`)
**{evaluation['std_population']:.6f}**, sample standard deviation (`ddof=1`)
{evaluation['std_sample']:.6f}, median {evaluation['median']:.3f}, minimum
{evaluation['minimum']:.0f}, maximum {evaluation['maximum']:.0f}.
Exactly **{evaluation['perfect_episodes']}/500 ({100*evaluation['success_fraction']:.2f}%)**
achieved 500; {evaluation['early_failures']} failed early. An approximate 95% Wilson
interval for success under this reset distribution is
[{100*(center_ci-radius):.2f}%, {100*(center_ci+radius):.2f}%]. This is strong measured
performance for this selected policy, not proof of perfect or universal robustness.
The final snapshot's separate 100-seed diagnostic mean was {final100['mean']:.3f},
with {final100['perfect_episodes']}/100 perfect episodes. That panel was not used for selection.

## Required figures and interpretation

All listed figures use actual saved data and are available as `.png` and `.pdf` in
`results/cartpole/{selection['run']}/figures/`.

1. **`initial_state_max_q`**: per episode, evaluate its initial observation using
   the online weights after that episode's updates, then take max over the two
   actions. The observed range was [{episodes.initial_state_max_q.min():.4f},
   {episodes.initial_state_max_q.max():.4f}], and the final point was
   {episodes.initial_state_max_q.iloc[-1]:.4f}. With gamma=.95, continuing unit rewards
   have scale 20, not 500. Finite 500-step discounted reward is
   `(1-.95**500)/.05`, effectively 20. Function approximation and transient
   overestimation can exceed this reference; it is not an enforced network bound.
2. **`training_rewards`**: undiscounted epsilon-greedy rewards, a strict trailing
   100-episode mean (first shown at episode 100), reward=500 reference and the
   separate greedy validation series. Exploration and changing network weights
   explain why training scores differ from fixed greedy evaluation. Final trailing
   training mean: {episodes.reward.tail(100).mean():.3f}. High isolated rewards
   are not sufficient evidence of stable convergence.
3. **`evaluation_histogram`**: exactly the 500 held-out results above, including
   mean, population SD and perfect-score count. It must not be confused with the
   periodic selection panel or the final-snapshot diagnostic.
4. **`policy_action_map`**: a non-standard 161x161 angle/angular-velocity action map,
   with contours of Q(right)-Q(left), fixing cart position and velocity to zero.
   Underlying values are in `metrics/policy_grid.csv`. This is a fixed slice of
   four-dimensional state space and not a complete policy representation.
5. **`initial_state_robustness`**: empirical return heatmap from 405 stress episodes,
   complementing the action map with actual dynamics. `optimization_diagnostics`
   shows mean Huber loss and epsilon; root `validation_learning_curves` compares runs.

Selected action-map probes (nearest grid points):

{probes}

Pushing toward a leaning/falling pole can move the cart underneath it. A simple
local heuristic is push right if `theta + .2*omega > 0`. On the central slice
`|theta|<=.05, |omega|<=.5`, the trained actions agree with this heuristic on
{100*agreement:.2f}% of points. This is an interpretive comparison, not a policy
correctness oracle. Read the probes and contours for asymmetry and unusual regions;
do not assume exact left/right symmetry or that extrapolated regions are safe.
Cart position and velocity also matter, and are suppressed by this visualization.
In particular, the sampled near-origin points are strongly right-biased: small
positive and negative angles at zero angular velocity both choose right. That
does not match a symmetric local corrective-control rule. The larger paired probes
(-.10,-1) and (+.10,+1) instead choose left and right respectively, consistent
with moving underneath a pole falling in the corresponding direction. These
observations qualify the physical interpretation: standard-reset success does
not establish sensible one-step actions throughout this slice. Closed-loop
trajectories, including the suppressed cart dimensions, remain the relevant test.

## Scientific validation: design and measured findings

All experiments use the same architecture, optimizer, replay/batch/warmup, epsilon
schedule, seed (except seed study), environment, validation panel, checkpoint
selection and evaluation seeds unless the row specifies a change. Each training
run is capped at 1,000 episodes and 80,000 steps; the comparison below reports actual
exposure. If a variant hits 1,000 episodes first, it has fewer updates, so its
performance comparison also reflects the episode-limited compute budget.

{performance}

`evaluation_std` is within-model population SD across 500 resets. The separate
stability table reports mean/P95 minibatch Huber loss, largest absolute sampled
Q-value, final initial-state max-Q and SD of periodic validation means:

{stability}

Raw loss and Q magnitudes depend on gamma and target dynamics; these are descriptive
diagnostics, not commensurate goodness scores. Periodic-score SD includes learning
trend and should not be interpreted as stationary variance. The comparison plot
uses environment steps rather than episode number to expose differing experience.

### A. Reproducibility / independent seeds

Hypothesis: identical settings need not produce identical learning outcomes, but
well-tuned DQN should frequently find a high-scoring policy. Changed: training/replay/
initialization/reset seed, 11 vs 22 vs 33. Held fixed: all algorithm settings and
evaluation seed panels. Completed {across['runs']} full runs and 500 evaluation episodes
per selected model. Mean of the per-run evaluation means:
**{across['mean_evaluation_reward']:.6f}**. Between-run population SD:
**{across['std_between_run_means_population']:.6f}**; between-run sample SD:
{across['std_between_run_means_sample']:.6f}. Mean success fraction:
{100*across['mean_success_fraction']:.3f}%. Fraction of runs achieving 500/500 perfect
test episodes: {across['fraction_runs_perfect_500_evaluation']:.3f}.
These are only three training seeds; they demonstrate substantial variability,
not a reliable across-seed convergence guarantee. The chosen primary seed's
performance is not representative of the mean training run.
Per-run test summaries were inspected as they completed, before the final primary
designation. That designation uses validation scores only and all seed results
are disclosed; no additional hyperparameter tuning followed test feedback.
An untouched confirmatory test panel would strengthen a future preregistered study.

### B. Target updates: hard, soft, and every-update synchronization

Hypothesis: target lag reduces moving-target instability; soft averaging may
smooth changes but changes effective lag. Changed: hard copies every 500 updates
versus soft `target <- .999*target + .001*online` after every update; an additional
ablation copies every update. Held fixed: seed 11 and all other settings.
The old-target coefficient follows Week 6 page 14 exactly. Soft .999 has a longer
typical lag than a hard-500 target, so the comparison is not a pure test of smoothness.
Measured held-out means were hard-500 {baseline.evaluation_mean:.3f}, soft
{soft.evaluation_mean:.3f}, every-update {every.evaluation_mean:.3f}; perfect-score
fractions were {baseline.success_fraction:.3f}, {soft.success_fraction:.3f},
{every.success_fraction:.3f}, respectively. Soft and every-update changed mean
reward by {soft.evaluation_mean-baseline.evaluation_mean:+.3f} and
{every.evaluation_mean-baseline.evaluation_mean:+.3f} versus hard-500.
The every-update run used {int(every.env_steps):,} steps and its largest sampled
absolute Q was {every.max_abs_q:.3f}. This directly tests whether removing useful
target lag helps in this setting. See the stability table and learning-curve panel. Differences
describe these one-seed, budget-limited runs; they do not establish universal superiority.
The every-update variant exhibits severe finite Q-value inflation, far above the
unit-reward discounted scale of 20, accompanied by large losses and nearly immediate
policy failures. Gradient clipping did not prevent this value instability. NaN/Inf
guards cannot detect all scientifically poor but finite estimates. Passing software
tests therefore does not imply successful algorithm behavior.

### C. Replay ablation

Hypothesis: temporally correlated batches reduce the diversity of Bellman updates
and may destabilize control. Changed: uniform replay vs the most recent 128
transitions, in chronological order, at every update. Replay capacity, batch size,
warmup and all remaining settings stay fixed at seed 11. Recent batches overlap
heavily across updates and are strongly correlated. This is not a no-memory method.
Recent-sample replay scored {recent.evaluation_mean:.3f} versus randomized replay's
{baseline.evaluation_mean:.3f}; success fractions were {recent.success_fraction:.3f}
versus {baseline.success_fraction:.3f}. The difference in held-out mean was
{recent.evaluation_mean-baseline.evaluation_mean:+.3f}. Its loss P95 was
{recent.loss_p95:.6f} versus {baseline.loss_p95:.6f}. Compare actual exposure and
curves before attributing the entire difference to correlation. This result does
not prove randomized replay is universally necessary. Any failure is an
algorithm/budget outcome, not automatically a bug.
The recent-sample variant has a lower mean return but a higher perfect-score
fraction than the random baseline, with larger within-model return variability.
This mixed result warns against ranking variants by loss alone or assuming the
ablation must agree with the expected advantage of randomized replay.

### D. Discount sensitivity

Hypothesis: gamma changes both effective planning horizon and Q scale, and may
change long-horizon pole/cart stabilization. Changed: gamma .90, .95, .99 at seed 11;
held fixed: remaining settings. Corresponding continuing unit-reward scales are
10, 20 and 100. Compare actual final Q values and reward performance in the tables.
Measured means at .90/.95/.99 were {g90.evaluation_mean:.3f}/
{baseline.evaluation_mean:.3f}/{g99.evaluation_mean:.3f}, with final initial-state
Q values {g90.final_initial_q:.3f}/{baseline.final_initial_q:.3f}/{g99.final_initial_q:.3f}.
Gamma=.95 remains the primary required model. Gamma=.99 can need more target
propagation/experience to approach its larger scale; a fixed-budget result is not
an asymptotic comparison, and lower loss need not imply better control.

### E. Ten-step and independent Bellman diagnostic

Hypothesis: greedy actions equal argmax of online Q, while detached targets use
the target network, genuine termination masks and the configured discount.
Fixed checkpoint and reset seed 31415; exactly {diagnostic['trajectory_steps']} steps
logged in `diagnostics/trajectory10.csv`, including both Q values, full state and next
state, action, reward, flags and independently calculated target. All
{diagnostic['greedy_action_matches']}/{diagnostic['trajectory_steps']} greedy actions
matched argmax. A float64 NumPy implementation independently recomputed target-network
linear/ReLU layers for {diagnostic['sampled_transitions']} sampled real transitions,
including {diagnostic['terminal_samples']} terminal and {diagnostic['truncated_samples']}
truncated transitions. Maximum absolute Bellman difference:
**{diagnostic['max_bellman_absolute_error']:.9g}**. Tolerances: atol 2e-5, rtol 2e-6.
The actual trained checkpoint reload also matched CUDA Q values with maximum error
{diagnostic['cuda_reload_max_absolute_error']:.9g}; CPU-vs-CUDA maximum error over
128 probe states was {diagnostic['cpu_reload_max_absolute_error']:.9g}, with
{100*diagnostic['cpu_gpu_greedy_agreement']:.2f}% greedy-action agreement. Tiny
cross-device differences need not be bitwise zero.
The diagnostic uses saved-checkpoint weights on historical sampled transitions;
it does not falsely claim to reproduce their historical training targets.
An independent analytic target with reward=1 and next max-Q=4 is 4.8 for both
ordinary and time-limited transitions, and 1.0 for genuine termination.

### F. Initial-state stress distribution

Hypothesis: strong performance near standard resets need not imply recovery
from large pole angle/angular velocity. Changed: angle in [-.18,.18] rad and
angular velocity in [-1.5,1.5] rad/s, on a 9x9 grid, five nuisance cart resets
per point (seeds 70000-70004). Initialization explicitly assigns
`env.unwrapped.state` after reset; dynamics, force, gravity and time step are
unchanged. This is not the standard CartPole reset distribution.
All {stress['episodes']} episodes completed: mean **{stress['mean']:.6f}**, population
SD {stress['std_population']:.6f}, perfect {stress['perfect_episodes']}/{stress['episodes']}
({100*stress['success_fraction']:.3f}%), minimum {stress['minimum']:.0f}.
The lowest grid mean was {low['mean']:.3f} at angle {low.theta_rad:.4f},
angular velocity {low.omega_rad_s:.4f}; highest {high['mean']:.3f} at
({high.theta_rad:.4f}, {high.omega_rad_s:.4f}). Extreme same-direction lean and
angular velocity can be physically unrecoverable. Failure there is not on its own
evidence of incorrect code. Only five nuisance states per point limits inference.

## Edge cases and implementation checks

The completed focused suite has 18 passing tests. `tests_complete.log` records the
result. Unit tests and the smoke experiment are distinguished from the above
algorithm-level experiments. They verify:

- Immediate failure using supported reset bounds outside the pole angle threshold:
  one step terminates and still contributes reward=1.
- A real 500-step episode under an analytic feedback controller: reward=500,
  truncated true, terminated false. This controller only tests bookkeeping and is
  never substituted for the learned DQN in reported policy evaluation.
- Analytical terminal/truncated Bellman masking, no target gradients, correct
  chosen-action indexing and independently known Huber-loss values.
- Hard target stays unchanged until its configured optimizer-update interval;
  soft target uses .999 on old weights and .001 on new online weights.
- Replay underflow raises an explicit error; wraparound retains the newest
  transitions; random sampling is diverse; recent sampling preserves chronology.
- Epsilon 0 always takes greedy action; epsilon 1 samples both actions at a plausible
  near-uniform frequency over 1,000 samples. The schedule is checked at global
  steps 0, 10,000, 20,000 and beyond.
- CPU checkpoint reload reproduces Q-values exactly. Evaluation cannot advance
  a separate training environment or mutate its weights.
- Nonfinite rewards/losses, online Q and target Q fail before optimizer update;
  gradient norm and updated parameters are also checked during training. Invalid
  configuration values fail clearly. No numerical failures were silently discarded.

The eight-episode CUDA smoke run used warmup 128 to exercise updates quickly.
This is a software check, not substantive learning evidence. An initial test
collection failure exposed that a nested venv did not inherit the parent venv's
site-packages; a project-only `.pth` file fixed package visibility without changing
shared installs. A preliminary SSH here-document had quoting/line-ending errors;
the subsequent file-based audit completed successfully. Neither failure is counted
as a failed algorithm experiment. Actual driver and subprocess exit codes must
be checked before interpreting results as complete.

## Comparison to course expectations and critical limitations

Replay, target lag and high-to-low exploration implement the Week 6 stabilization
mechanisms; Huber loss and clipping follow page 20's recommendations. The lab's
semi-gradient target is detached rather than differentiated through the target.
The lecture warns that runs can be inconsistent and loss need not decrease
monotonically; measured seed variability and transient validation drops support
that caution. Declining regression loss alone would not prove successful control.

The initial seed-11 baseline's selected mean was {baseline.evaluation_mean:.3f},
success {100*baseline.success_fraction:.1f}%, while its final-snapshot diagnostic
mean was {baseline.final_snapshot_mean:.3f} on a different 100-seed panel. This is
consistent with unstable learning, though the panels are not paired. Preserve
successful checkpoints rather than assuming the last model is best.

Strengths are an explicit inspectable algorithm, correct time-limit semantics,
measured single-GPU execution, held-out evaluation, independent numerical references,
paired test seeds across variants and honest disclosure of negative results.
Limitations include three baseline seeds but only one seed per ablation, a small
reused selection panel, budget/episode ceilings, no exhaustive hyperparameter search,
no full training resume, and action slices that hide two dimensions. The selected
primary policy performs well in its tested reset distribution; neither perfect
control nor robust learning across seeds is established.

## Additional three months

Month 1: increase to 20-30 training seeds; preregister a hyperparameter grid and
validation protocol; use rotating validation panels and one untouched final test;
report confidence intervals, paired per-reset differences, sample efficiency and
run-to-run success probabilities. Match environment-step and optimizer budgets
explicitly even when some variants need more than 1,000 episodes. Repeatedly
evaluate a frozen checkpoint to quantify implementation/hardware repeatability.

Month 2: match mean target lag before comparing hard and soft updates; sweep
target interval/tau, replay capacity, batch size, learning rate and exploration
separately and factorially. Add Double DQN only as an explicitly named extension
to test overestimation, keeping Classic DQN as the reference. Compare predicted
Q with independently estimated discounted Monte Carlo returns, including terminal
and time-limit handling, and assess calibration rather than only Q magnitude.

Month 3: map the full four-dimensional initial-state recovery region with
adaptive sampling; analyze cart drift and failure trajectories; stress observation
noise, sensor delay and physical parameters in explicitly separate environments;
compare local learned feedback to a linearized control baseline and estimate
recoverability bounds. Test CPU/GPU and cross-version numerical reproducibility,
complete resumable checkpoints, and replicate with independent implementations.
Publish raw data and negative results, then write the final PDF with clear limitations.

## Exact reproduction commands

Execute from the workspace root on deepdish5; inspect GPU availability and configure
the environment as in README first. Select an idle UUID anew, rather than assuming
this session's card remains available. Existing run/evaluation directories intentionally
cause an error: use a fresh results root or names when reproducing.

```bash
export CUDA_VISIBLE_DEVICES=GPU-cef65388-88b0-f39a-6bac-c5316eb78540
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
.venv/bin/python -m compileall -q assignment2
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest assignment2/tests -q
'''
    tail='\n'.join(commands)+'''
# After each run, exact-500 evaluation and plots (run names substituted above):
.venv/bin/python -m assignment2.cartpole.validate finalize --run assignment2/results/cartpole/main_seed11
# Or execute all seven validation runs and their evaluations sequentially:
.venv/bin/python -m assignment2.cartpole.validate suite --root assignment2/results/cartpole --max-steps 80000
.venv/bin/python -m assignment2.cartpole.validate primary
.venv/bin/python -m assignment2.cartpole.validate diagnostics --run assignment2/results/cartpole/seed33
.venv/bin/python -m assignment2.cartpole.validate robustness --run assignment2/results/cartpole/seed33
.venv/bin/python -m assignment2.cartpole.validate compare
.venv/bin/python assignment2/build_report.py
```

Individual final evaluation/plot commands for the selected primary policy:

```bash
.venv/bin/python -m assignment2.cartpole.evaluate --checkpoint assignment2/results/cartpole/seed33/checkpoints/best_ep0942.pt --output assignment2/results/cartpole/seed33/evaluation500
.venv/bin/python -m assignment2.cartpole.plots --run assignment2/results/cartpole/seed33 --checkpoint assignment2/results/cartpole/seed33/checkpoints/best_ep0942.pt --evaluation assignment2/results/cartpole/seed33/evaluation500 --device cuda:0
```

Do not execute both the individual training commands and the suite into the same
paths. These are alternative reproduction routes. Per-run command ledgers contain
absolute interpreter paths, elapsed seconds and return codes. The initial main
command is additionally recorded in `run_main.sh` and `main_driver.log`.

## Remaining submission work

No required CartPole scientific experiment is marked complete unless its output
exists. All eight configurations in the comparison table completed training,
selected-checkpoint 500-episode evaluation and figures. Baseline and primary
numerical/stress diagnostics are saved. Further multi-seed ablations, full-resume
support, repository publication and a final assembled submission PDF remain outside
these completed measurements. GitHub username/repository/URL must be supplied by
the author. MsPacman is intentionally unimplemented.
'''
    (ROOT/'REPORT_NOTES_CARTPOLE.md').write_text(header+tail,encoding='utf-8')
    # A brief results pointer keeps README concise and avoids duplicate hand-entered numbers.
    (RESULTS/'RESULTS_SUMMARY.md').write_text(
        f"# Measured CartPole results\n\nPrimary: `{selection['run']}`; checkpoint `{selection['checkpoint']}`.\n\n"
        f"500 greedy episodes: mean {evaluation['mean']:.6f}, population SD {evaluation['std_population']:.6f}; "
        f"{evaluation['perfect_episodes']}/500 perfect.\n\n"+performance+'\n',encoding='utf-8')
    print(ROOT/'REPORT_NOTES_CARTPOLE.md')


if __name__=='__main__':report()
