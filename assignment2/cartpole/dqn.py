import numpy as np
import torch
from torch import nn


def finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite {name}; stop run and inspect transitions, learning rate and checkpoint')


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, state):
        return self.layers(state)


@torch.no_grad()
def bellman_target(target, next_state, reward, terminated, gamma):
    q = target(next_state)
    finite(q, 'target Q-values')
    return reward + gamma * (~terminated).float() * q.max(dim=1).values


class DQN:
    def __init__(self, config, device='cpu'):
        self.config, self.device = config, torch.device(device)
        self.online = QNetwork().to(device)
        self.target = QNetwork().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.updates = 0
        self.target_updates = 0

    @torch.no_grad()
    def q_values(self, states):
        q = self.online(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device))
        finite(q, 'online Q-values')
        return q.cpu().numpy()

    def act(self, state, epsilon, rng):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon outside [0,1]')
        if rng.random() < epsilon:
            return int(rng.integers(2))
        return int(self.q_values(state).argmax())

    def update(self, batch):
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        q = self.online(b['state']).gather(1, b['action'][:, None]).squeeze(1)
        finite(q, 'online Q-values')
        targets = bellman_target(self.target, b['next_state'], b['reward'], b['terminated'], self.config.gamma)
        loss = nn.functional.smooth_l1_loss(q, targets)
        self.last_update_devices = dict(states=str(b['state'].device), selected_q=str(q.device),
                                        targets=str(targets.device), loss=str(loss.device))
        finite(loss, 'Huber loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        for p in self.online.parameters():
            finite(p, 'updated network weights')
        self.updates += 1
        if self.config.target_mode == 'soft':
            with torch.no_grad():
                for target, online in zip(self.target.parameters(), self.online.parameters()):
                    target.mul_(self.config.target_tau).add_(online, alpha=1-self.config.target_tau)
            self.target_updates += 1
        elif self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_updates += 1
        return float(loss.detach()), float(norm), float(q.detach().abs().max())
