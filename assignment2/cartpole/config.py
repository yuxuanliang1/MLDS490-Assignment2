from dataclasses import dataclass, asdict
import argparse
import random
import numpy as np
import torch


@dataclass
class Config:
    env_id: str = 'CartPole-v1'
    gamma: float = 0.95
    seed: int = 11
    episodes: int = 1000
    capacity: int = 50000
    warmup: int = 1000
    batch_size: int = 128
    learning_rate: float = 5e-4
    target_interval: int = 500
    target_mode: str = 'hard'
    target_tau: float = 0.999
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_steps: int = 20000
    grad_clip: float = 10.0
    replay_mode: str = 'random'
    eval_every: int = 25
    eval_episodes: int = 10
    early_stop_checks: int = 0
    early_stop_score: float = 500.0
    max_steps: int = 300000

    def __post_init__(self):
        if self.env_id != 'CartPole-v1':
            raise ValueError('Only CartPole-v1 is supported')
        if not 0 <= self.gamma < 1:
            raise ValueError('gamma must be in [0,1)')
        for k in ('episodes', 'capacity', 'warmup', 'batch_size', 'target_interval',
                  'epsilon_steps', 'eval_every', 'eval_episodes', 'max_steps'):
            if getattr(self, k) <= 0:
                raise ValueError(f'{k} must be positive')
        if self.capacity < self.batch_size or self.warmup < self.batch_size:
            raise ValueError('capacity and warmup must be >= batch_size')
        if self.capacity < self.warmup:
            raise ValueError('capacity must be >= warmup')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('invalid epsilon bounds')
        if self.replay_mode not in ('random', 'recent'):
            raise ValueError('invalid replay mode')
        if self.target_mode not in ('hard', 'soft') or not 0 <= self.target_tau < 1:
            raise ValueError('invalid target mode/tau; tau weights OLD target')
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError('learning_rate and grad_clip must be positive')

    def epsilon(self, step):
        fraction = min(max(step, 0) / self.epsilon_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def add_config_args(parser):
    for key, value in asdict(Config()).items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)


def config_from_args(args):
    return Config(**{k: getattr(args, k) for k in asdict(Config())})
