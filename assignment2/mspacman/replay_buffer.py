"""Compact preallocated image replay, including independently retained next images.

Storing both images avoids reconstruction errors across episodes/wraparound. At
50,000 transitions it needs ~705 MB, feasible after checking host available RAM.
"""
from pathlib import Path
import json
import os
import numpy as np
import psutil


class ReplayBuffer:
    def __init__(self,capacity,seed=0,check_memory=True):
        if capacity<=0:raise ValueError('capacity must be positive')
        self.capacity=capacity;self.position=0;self.size=0
        self.rng=np.random.default_rng(seed)
        self.estimated_bytes=self.memory_bytes(capacity)
        if check_memory and self.estimated_bytes>psutil.virtual_memory().available*.25:
            raise MemoryError(f'Replay needs {self.estimated_bytes/2**30:.2f} GiB; exceeds 25% of available host RAM')
        try:
            self.data=dict(state=np.empty((capacity,1,88,80),np.int8),
                next_state=np.empty((capacity,1,88,80),np.int8),action=np.empty(capacity,np.int64),
                reward=np.empty(capacity,np.float32),terminated=np.empty(capacity,bool),truncated=np.empty(capacity,bool))
        except MemoryError as error:raise MemoryError('Replay allocation failed; reduce --capacity') from error

    @staticmethod
    def memory_bytes(capacity):return capacity*(2*88*80+8+4+1+1)

    def __len__(self):return self.size

    def add(self,state,action,reward,next_state,terminated,truncated):
        if not isinstance(action,(int,np.integer)) or not 0<=action<9:raise ValueError('Action must be integer in 0..8')
        if not np.isfinite(reward):raise FloatingPointError('Nonfinite replay reward')
        for image in (state,next_state):
            if image.shape!=(1,88,80) or image.dtype!=np.int8:raise ValueError('Replay requires int8 CHW images')
        for key,value in dict(state=state,next_state=next_state,action=action,reward=reward,
                              terminated=terminated,truncated=truncated).items():self.data[key][self.position]=value
        self.position=(self.position+1)%self.capacity;self.size=min(self.size+1,self.capacity)

    def sample(self,count,mode='random'):
        if not 0<count<=self.size:raise ValueError(f'Underflow: requested {count}, have {self.size}')
        if mode=='random':indices=self.rng.choice(self.size,count,replace=False)
        elif mode=='recent':indices=(self.position-np.arange(count,0,-1))%self.capacity
        else:raise ValueError(mode)
        return {key:value[indices] for key,value in self.data.items()}

    def save(self,path):
        path=Path(path);temporary=path.with_suffix('.tmp.npz')
        # Slices are views; no full-array float conversion or duplicate allocation.
        np.savez_compressed(temporary,**{key:value[:self.size] for key,value in self.data.items()},
            metadata=np.array(json.dumps(dict(capacity=self.capacity,size=self.size,position=self.position,
                                              rng=self.rng.bit_generator.state))))
        os.replace(temporary,path)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            buffer=cls(metadata['capacity']);buffer.size=metadata['size'];buffer.position=metadata['position']
            for key in buffer.data:buffer.data[key][:buffer.size]=archive[key]
            buffer.rng.bit_generator.state=metadata['rng']
        return buffer
