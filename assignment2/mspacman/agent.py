from dataclasses import asdict
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import Config
from .model import QNetwork


def finite(value,name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {name}; inspect rewards, learning rate, target lag and last checkpoint')


@torch.no_grad()
def bellman_target(target,next_states,rewards,terminated,gamma):
    q=target(next_states);finite(q,'target Q-values')
    result=rewards+gamma*(~terminated).float()*q.max(1).values
    finite(result,'Bellman targets')
    return result


class DQN:
    def __init__(self,config,device='cpu'):
        self.config=config;self.device=torch.device(device)
        self.online=QNetwork().to(self.device);self.target=QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict());self.target.requires_grad_(False);self.target.eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config.learning_rate)
        self.updates=0;self.target_updates=0

    @torch.no_grad()
    def q_values(self,states):
        states=np.asarray(states)
        if states.ndim==3:states=states[None]
        if states.dtype!=np.int8:raise ValueError('CNN inference requires compact signed int8 images')
        q=self.online(torch.as_tensor(states,device=self.device));finite(q,'online inference Q-values')
        return q.cpu().numpy()

    def act(self,state,epsilon,rng):
        if not 0<=epsilon<=1:raise ValueError('epsilon outside [0,1]')
        return int(rng.integers(9)) if rng.random()<epsilon else int(self.q_values(state).argmax())

    def update(self,batch):
        if not ((batch['action']>=0)&(batch['action']<9)).all():raise ValueError('Invalid replay action')
        b={key:torch.as_tensor(value,device=self.device) for key,value in batch.items()}
        all_q=self.online(b['state']);finite(all_q,'online training Q-values')
        q=all_q.gather(1,b['action'][:,None]).squeeze(1)
        targets=bellman_target(self.target,b['next_state'],b['reward'],b['terminated'],self.config.gamma)
        loss=nn.functional.smooth_l1_loss(q,targets);finite(loss,'Huber loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(self.online.parameters(),self.config.grad_clip,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        if self.updates%self.config.target_interval==0:
            self.target.load_state_dict(self.online.state_dict());self.target_updates+=1
        if any(p.grad is not None for p in self.target.parameters()):raise RuntimeError('Unexpected target-network gradient')
        self.last_update_devices=dict(input=str(b['state'].device),q=str(q.device),target=str(targets.device),
            loss=str(loss.device),gradient=str(next(self.online.parameters()).grad.device),
            adam_moments=str(self.optimizer.state[next(self.online.parameters())]['exp_avg'].device))
        return dict(loss=float(loss.detach()),grad_norm=float(norm),mean_q=float(all_q.detach().mean()),
                    max_abs_q=float(all_q.detach().abs().max()),target_updates=self.target_updates)


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return value


def save_model(path,agent,**metadata):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    state=dict(kind='model_only',config=asdict(agent.config),online=cpu_tree(agent.online.state_dict()),
        target=cpu_tree(agent.target.state_dict()),updates=agent.updates,target_updates=agent.target_updates,**metadata)
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,path)


def load_agent(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    agent=DQN(Config(**checkpoint['config']),device)
    agent.online.load_state_dict(checkpoint['online']);agent.target.load_state_dict(checkpoint['target'])
    agent.updates=checkpoint['updates'];agent.target_updates=checkpoint['target_updates']
    return agent,checkpoint
