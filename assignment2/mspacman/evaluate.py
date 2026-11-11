import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .agent import load_agent
from .environment import make_env
from .preprocessing import preprocess_chw
from .runtime import gpu_audit,write_json,seed_all


def evaluate_policy(agent,seeds,batch_envs=16):
    rows=[];started=time.monotonic()
    for offset in range(0,len(seeds),batch_envs):
        selected=seeds[offset:offset+batch_envs];envs=[make_env(int(seed)) for seed in selected]
        try:
            states=[]
            for env,seed in zip(envs,selected):states.append(preprocess_chw(env.reset(seed=int(seed))[0]))
            active=list(range(len(selected)));scores=np.zeros(len(selected));lengths=np.zeros(len(selected),int)
            frames=np.zeros(len(selected),int);life_losses=np.zeros(len(selected),int)
            lives=[env.unwrapped.ale.lives() for env in envs]
            while active:
                actions=agent.q_values(np.stack([states[i] for i in active])).argmax(1)
                keep=[]
                for i,action in zip(active,actions):
                    before=envs[i].unwrapped.ale.getFrameNumber()
                    obs,reward,term,trunc,info=envs[i].step(int(action))
                    frames[i]+=envs[i].unwrapped.ale.getFrameNumber()-before
                    current_lives=envs[i].unwrapped.ale.lives()
                    life_losses[i]+=int(current_lives<lives[i]);lives[i]=current_lives
                    states[i]=preprocess_chw(obs);scores[i]+=reward;lengths[i]+=1
                    if term or trunc:
                        rows.append(dict(seed=int(selected[i]),reward=float(scores[i]),length=int(lengths[i]),
                            raw_frames=int(frames[i]),terminated=bool(term),truncated=bool(trunc),life_losses=int(life_losses[i])))
                    else:keep.append(i)
                active=keep
        finally:
            for env in envs:env.close()
        if len(seeds)>=100:print(f'evaluation={len(rows)}/{len(seeds)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    return pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)


def summarize(data):
    r=data.reward.to_numpy()
    return dict(episodes=len(data),mean=float(r.mean()),std_population=float(r.std(ddof=0)),
        median=float(np.median(r)),minimum=float(r.min()),maximum=float(r.max()),
        mean_decision_steps=float(data.length.mean()),mean_raw_frames=float(data.raw_frames.mean()),
        terminated_episodes=int(data.terminated.sum()),truncated_episodes=int(data.truncated.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed-start',type=int,default=900000);p.add_argument('--batch-envs',type=int,default=16)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'gpu_audit.json',gpu_audit());agent,checkpoint=load_agent(a.checkpoint,'cuda:0')
    seed_all(agent.config.seed);start=time.monotonic()
    data=evaluate_policy(agent,list(range(a.seed_start,a.seed_start+500)),a.batch_envs)
    assert len(data)==data.seed.nunique()==500
    data.to_csv(out/'episodes.csv',index=False)
    result=summarize(data);result.update(checkpoint=a.checkpoint,checkpoint_episode=checkpoint['episode'],
        epsilon=0.,seed_start=a.seed_start,elapsed_seconds=time.monotonic()-start)
    write_json(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
