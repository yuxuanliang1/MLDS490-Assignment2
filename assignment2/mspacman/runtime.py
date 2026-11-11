"""Atari-specific audit/serialization; CartPole utilities are reused read-only."""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from assignment2.cartpole.runtime import gpu_audit


def write_json(path, value):
    def convert(item):
        if isinstance(item,dict):return {str(key):convert(val) for key,val in item.items()}
        if isinstance(item,(list,tuple)):return [convert(val) for val in item]
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,Path):return str(item)
        return item
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(convert(value),allow_nan=False,indent=2)+'\n')
    os.replace(temporary,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def rng_state():
    np_state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[np_state[0],np_state[1].tolist(),*np_state[2:]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np_state=state['numpy'];np.random.set_state((np_state[0],np.array(np_state[1],np.uint32),*np_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
