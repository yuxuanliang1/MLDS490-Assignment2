import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from .config import Config, seed_all
from .dqn import DQN
from .runtime import gpu_audit, write_json


def load_agent(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config(**checkpoint['config'])
    agent = DQN(config, device)
    agent.online.load_state_dict(checkpoint['online'])
    agent.target.load_state_dict(checkpoint['target'])
    agent.online.eval()
    return agent, checkpoint


def evaluate_policy(agent, seeds, batch_envs=50, reset_options=None):
    """Batch network inference only; each environment stops on either end flag."""
    rows = []
    for start in range(0, len(seeds), batch_envs):
        chunk = seeds[start:start + batch_envs]
        envs = [gym.make('CartPole-v1') for _ in chunk]
        try:
            states = [env.reset(seed=int(seed), options=reset_options)[0] for env, seed in zip(envs, chunk)]
            active = list(range(len(chunk)))
            rewards = np.zeros(len(chunk))
            lengths = np.zeros(len(chunk), dtype=int)
            while active:
                q = agent.q_values(np.array([states[i] for i in active]))
                next_active = []
                for i, action in zip(active, q.argmax(axis=1)):
                    states[i], reward, terminated, truncated, _ = envs[i].step(int(action))
                    rewards[i] += reward
                    lengths[i] += 1
                    if terminated or truncated:
                        rows.append(dict(seed=int(chunk[i]), reward=float(rewards[i]),
                                         length=int(lengths[i]), terminated=bool(terminated), truncated=bool(truncated)))
                    else:
                        next_active.append(i)
                active = next_active
        finally:
            for env in envs:
                env.close()
    return pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)


def summarize(frame):
    rewards = frame.reward.to_numpy()
    return dict(episodes=len(rewards), mean=float(rewards.mean()),
                std_population=float(rewards.std(ddof=0)), std_sample=float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.,
                perfect_episodes=int((rewards == 500).sum()), success_fraction=float((rewards == 500).mean()),
                median=float(np.median(rewards)), early_failures=int((rewards < 500).sum()),
                minimum=float(rewards.min()), maximum=float(rewards.max()))


def main():
    parser = argparse.ArgumentParser(description='Exactly 500 held-out greedy episodes')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed-start', type=int, default=100000)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'gpu_audit.json', gpu_audit())
    agent, checkpoint = load_agent(args.checkpoint, 'cuda:0')
    seed_all(agent.config.seed)
    frame = evaluate_policy(agent, list(range(args.seed_start, args.seed_start + 500)))
    assert len(frame) == 500 and frame.seed.nunique() == 500
    frame.to_csv(output / 'episodes.csv', index=False)
    summary = summarize(frame)
    summary.update(checkpoint=str(args.checkpoint), checkpoint_episode=checkpoint['episode'],
                   epsilon=0., seed_start=args.seed_start,
                   selection_reason='Highest periodic greedy mean; earliest checkpoint wins ties; held-out seeds unused in selection')
    write_json(output / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
