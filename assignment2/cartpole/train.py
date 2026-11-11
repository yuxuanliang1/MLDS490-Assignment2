import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from .config import add_config_args, config_from_args, seed_all
from .dqn import DQN
from .evaluate import evaluate_policy, summarize
from .replay_buffer import ReplayBuffer
from .runtime import gpu_audit, write_json


def save_checkpoint(path, agent, episode, steps, score):
    # CPU tensors make the model portable. These are inference checkpoints, not exact-resume snapshots.
    torch.save(dict(config=asdict(agent.config),
                    online={k: v.detach().cpu() for k, v in agent.online.state_dict().items()},
                    target={k: v.detach().cpu() for k, v in agent.target.state_dict().items()},
                    episode=episode, env_steps=steps, optimizer_updates=agent.updates,
                    target_updates=agent.target_updates, eval_mean=score), path)


def train(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite any previous experiment.
    for sub in ('logs', 'metrics', 'checkpoints', 'figures'):
        (output / sub).mkdir()
    write_json(output / 'metrics/gpu_audit.json', gpu_audit())
    write_json(output / 'metrics/config.json', asdict(config))
    seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    agent = DQN(config, 'cuda:0')
    replay = ReplayBuffer(config.capacity, config.seed)
    env = gym.make(config.env_id)
    assert env.spec.max_episode_steps == 500
    env.action_space.seed(config.seed)
    t0 = time.monotonic()
    steps, best, streak, last_eval = 0, -float('inf'), 0, None
    stop_reason = 'max_episodes'
    reward_history = deque(maxlen=100)
    fields = ['episode','reward','length','epsilon','mean_loss','mean_grad_norm','target_updates','trailing_100_mean',
              'max_abs_batch_q','initial_state_max_q','env_steps','optimizer_updates','elapsed_seconds','terminated','truncated']
    try:
        with (output / 'logs/episodes.csv').open('w', newline='') as ef, \
             (output / 'logs/updates.csv').open('w', newline='') as uf, \
             (output / 'logs/transitions.csv').open('w', newline='') as tf, \
             (output / 'logs/periodic_evaluation.csv').open('w', newline='') as vf:
            ew = csv.DictWriter(ef, fieldnames=fields); ew.writeheader()
            uw = csv.writer(uf); uw.writerow(['update','env_step','loss','grad_norm','max_abs_q'])
            tw = csv.writer(tf); tw.writerow(['episode','env_step','x','x_dot','theta','theta_dot','action','reward','next_x','next_x_dot','next_theta','next_theta_dot','terminated','truncated'])
            vw = csv.writer(vf); vw.writerow(['episode','mean','std_population','success_fraction'])
            for episode in range(1, config.episodes + 1):
                state, _ = env.reset(seed=config.seed + episode - 1)
                initial = state.copy()
                reward_sum, length = 0., 0
                losses, norms, qs = [], [], []
                while True:
                    epsilon = config.epsilon(steps)
                    action = agent.act(state, epsilon, rng)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    steps += 1; length += 1; reward_sum += reward
                    replay.add(state, action, reward, next_state, terminated, truncated)
                    tw.writerow([episode, steps, *state, action, reward, *next_state, terminated, truncated])
                    state = next_state
                    if len(replay) >= config.warmup:
                        loss, norm, max_q = agent.update(replay.sample(config.batch_size, config.replay_mode))
                        losses.append(loss); norms.append(norm); qs.append(max_q)
                        uw.writerow([agent.updates, steps, loss, norm, max_q])
                    if terminated or truncated:
                        break
                reward_history.append(reward_sum)
                row = dict(episode=episode, reward=reward_sum, length=length, epsilon=epsilon,
                           target_updates=agent.target_updates,
                           trailing_100_mean=float(np.mean(reward_history)) if len(reward_history)==100 else '',
                           mean_loss=float(np.mean(losses)) if losses else '',
                           mean_grad_norm=float(np.mean(norms)) if norms else '',
                           max_abs_batch_q=max(qs) if qs else '',
                           initial_state_max_q=float(agent.q_values(initial).max()),
                           env_steps=steps, optimizer_updates=agent.updates, elapsed_seconds=time.monotonic()-t0,
                           terminated=terminated, truncated=truncated)
                ew.writerow(row); ef.flush(); uf.flush(); tf.flush()
                should_eval = episode % config.eval_every == 0 or episode == config.episodes or steps >= config.max_steps
                if should_eval:
                    # Fixed selection panel, separate from held-out final evaluation.
                    panel = evaluate_policy(agent, list(range(50000, 50000 + config.eval_episodes)))
                    result = summarize(panel)
                    last_eval = result['mean']
                    vw.writerow([episode,result['mean'],result['std_population'],result['success_fraction']]); vf.flush()
                    if last_eval > best:
                        best = last_eval
                        # Keep every improving checkpoint, including previous successes.
                        filename = f'best_ep{episode:04d}.pt'
                        save_checkpoint(output / 'checkpoints' / filename, agent, episode, steps, best)
                        write_json(output / 'metrics/selection.json', dict(checkpoint=filename, episode=episode,
                            eval_mean=best, selection_seed_start=50000, selection_episodes=config.eval_episodes,
                            reason='Highest periodic greedy mean; earliest wins ties; no held-out evaluation used'))
                    streak = streak + 1 if last_eval >= config.early_stop_score else 0
                    print(f'episode={episode} steps={steps} reward={reward_sum:.0f} eval={last_eval:.2f} best={best:.2f} elapsed={time.monotonic()-t0:.1f}s', flush=True)
                if config.early_stop_checks and streak >= config.early_stop_checks:
                    stop_reason = 'periodic_greedy_early_stop'; break
                if steps >= config.max_steps:
                    stop_reason = 'max_steps_at_episode_boundary'; break
            save_checkpoint(output / 'checkpoints/final.pt', agent, episode, steps, last_eval)
            write_json(output / 'metrics/training_summary.json', dict(episodes=episode, env_steps=steps,
                optimizer_updates=agent.updates, elapsed_seconds=time.monotonic()-t0, best_periodic_mean=best,
                target_updates=agent.target_updates, model_device=str(next(agent.online.parameters()).device),
                target_device=str(next(agent.target.parameters()).device),
                gradient_device=str(next(agent.online.parameters()).grad.device) if agent.updates else None,
                optimizer_tensor_devices=sorted({str(v.device) for s in agent.optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor)}),
                last_update_devices=agent.last_update_devices if agent.updates else {},
                stop_reason=stop_reason, max_steps_note='Budget checked at episode boundary; overshoot <=499 steps'))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train(config_from_args(args), args.output)


if __name__ == '__main__':
    main()
