"""Read-only environment/GPU preflight; no training or package mutation."""
import importlib.metadata as metadata
import argparse
import inspect
import json
from pathlib import Path
import time
import ale_py
import gymnasium as gym
import numpy as np
from .runtime import gpu_audit, write_json
from .environment import make_env,describe
from .preprocessing import preprocess_chw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='assignment2/results/mspacman/preflight_complete')
    args=parser.parse_args();output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    audit=gpu_audit()
    audit['packages']={name:metadata.version(name) for name in ['ale-py','psutil','pytest']}
    gym.register_envs(ale_py)
    print('MsPacman registrations:',[key for key in gym.registry if 'MsPacman' in key],flush=True)
    spec=gym.spec('ALE/MsPacman-v5')
    print(spec,flush=True)
    print(inspect.signature(ale_py.env.AtariEnv),flush=True)
    env=make_env(123)
    obs,info=env.reset(seed=123)
    audit['environment']=describe(env)
    processed=preprocess_chw(obs)
    audit['preprocessing']=dict(shape=processed.shape,dtype=str(processed.dtype),minimum=processed.min(),maximum=processed.max())
    start=time.monotonic(); reward_sum=0.;length=0;endings=[];lives=[]
    rng=np.random.default_rng(123)
    raw_frames=0
    for step in range(10000):
        before=env.unwrapped.ale.getFrameNumber()
        obs,reward,term,trunc,info=env.step(int(rng.integers(env.action_space.n)))
        raw_frames+=env.unwrapped.ale.getFrameNumber()-before
        reward_sum+=reward;length+=1
        if term or trunc:
            endings.append(dict(steps=length,score=reward_sum,terminated=term,truncated=trunc,info=info))
            if len(endings)==3:break
            obs,info=env.reset(seed=123+len(endings));reward_sum=0.;length=0
    audit['random_rollout']=dict(steps=step+1,frames=raw_frames,elapsed_seconds=time.monotonic()-start,episodes=endings)
    audit['rom_note']='ROM already bundled/available in installed ALE; no ROM download or license acceptance performed'
    env.close()
    write_json(output/'audit.json',audit)
    print((output/'audit.json').read_text(),flush=True)


if __name__=='__main__':main()
