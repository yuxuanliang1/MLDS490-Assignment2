"""Fixed allocation, uniformly sampled replay; both Gymnasium end flags survive."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=0):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.data = {
            'state': np.empty((capacity, 4), np.float32),
            'action': np.empty(capacity, np.int64),
            'reward': np.empty(capacity, np.float32),
            'next_state': np.empty((capacity, 4), np.float32),
            'terminated': np.empty(capacity, np.bool_),
            'truncated': np.empty(capacity, np.bool_),
        }

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated, truncated):
        values = (state, action, reward, next_state, terminated, truncated)
        if not np.isfinite(np.asarray(state)).all() or not np.isfinite(np.asarray(next_state)).all() or not np.isfinite(reward):
            raise FloatingPointError('Nonfinite transition: inspect environment state/reward')
        for key, value in zip(self.data, values):
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, mode='random'):
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError(f'Need {batch_size} transitions, have {self.size}')
        if mode == 'random':
            indices = self.rng.choice(self.size, batch_size, replace=False)
        elif mode == 'recent':
            indices = (self.position - np.arange(batch_size, 0, -1)) % self.capacity
        else:
            raise ValueError(mode)
        return {k: v[indices].copy() for k, v in self.data.items()}
