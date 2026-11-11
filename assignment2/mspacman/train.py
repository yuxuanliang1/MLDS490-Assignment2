import argparse
from collections import deque
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
import signal
import time
import traceback
import zlib
import numpy as np
import psutil
import torch
from .agent import DQN,save_model
from .checkpoint import save_recovery,load_recovery,rollback_logs
from .config import add_args,from_args
from .environment import make_env,describe
from .evaluate import evaluate_policy,summarize
from .preprocessing import preprocess_chw
from .replay_buffer import ReplayBuffer
from .runtime import gpu_audit,write_json,seed_all


def create_references(run,count):
    env=make_env(410000);rng=np.random.default_rng(410000);obs,_=env.reset(seed=410000)
    raw=[];processed=[];reset=0
    try:
        for step in range(count*25):
            if step%25==0:raw.append(obs.copy());processed.append(preprocess_chw(obs))
            obs,_,term,trunc,_=env.step(int(rng.integers(9)))
            if term or trunc:
                reset+=1;obs,_=env.reset(seed=410000+reset)
    finally:env.close()
    np.save(run/'metrics/reference_images.npy',np.stack(processed))
    np.save(run/'metrics/reference_rgb.npy',np.stack(raw))
    return np.stack(processed)


def train(config,output,resume=None):
    run=Path(output)
    run.mkdir(parents=True,exist_ok=resume is not None)
    if not resume:
        for sub in ['checkpoints','replay','figures','metrics','logs']:(run/sub).mkdir()
    audit=gpu_audit();audit['ale_py']=importlib.metadata.version('ale-py')
    session=len(list((run/'metrics').glob('gpu_audit_session*.json')))
    suffix=f'_resume{session:02d}' if resume else ''
    write_json(run/'metrics'/f'gpu_audit_session{session:02d}.json',audit)
    write_json(run/'metrics'/f'effective_config_session{session:02d}.json',dict(config=asdict(config),resume=resume))
    seed_all(config.seed);behavior=np.random.default_rng(config.seed)
    if resume:
        agent,replay,restored=load_recovery(resume,run,'cuda:0')
        old=asdict(agent.config);new=asdict(config)
        for key in old:
            if key not in ['episodes','max_steps','max_hours'] and old[key]!=new[key]:raise ValueError(f'Resume cannot silently change {key}')
        agent.config=config;behavior.bit_generator.state=restored['behavior_rng']
        completed,steps,frames,elapsed_base,best=(restored[k] for k in ['episode','steps','frames','elapsed_seconds','best_validation'])
        refs=np.load(run/'metrics/reference_images.npy');mode='a'
        rollback_logs(run,restored,session)
        # Reconstruct moving-average history without changing training RNG.
        import pandas as pd
        history=deque(pd.read_csv(run/'logs/episodes.csv').reward.tail(100),maxlen=100)
        if completed>=config.episodes or steps>=config.max_steps:raise ValueError('Resume requires a larger episode/step budget')
    else:
        agent=DQN(config,'cuda:0');replay=ReplayBuffer(config.capacity,config.seed)
        completed=steps=frames=0;elapsed_base=0.;best=-float('inf');mode='w';history=deque(maxlen=100)
        refs=create_references(run,config.reference_count)
        write_json(run/'metrics/config.json',asdict(config))
        save_model(run/'checkpoints/initial.pt',agent,episode=0,steps=0,frames=0)
    reference_sha=hashlib.sha256((run/'metrics/reference_images.npy').read_bytes()).hexdigest()
    if resume and reference_sha!=restored['reference_sha256']:raise RuntimeError('Fixed reference images changed across resume')
    write_json(run/'metrics/reference_protocol.json',dict(sha256=reference_sha,count=len(refs),
        collection_seed=410000,random_decision_interval=25,unique_images=len(np.unique(refs.reshape(len(refs),-1),axis=0)),
        statistic='Mean over fixed references of max over actions Q_online; population SD across references also logged'))
    env=make_env(config.seed);write_json(run/'metrics/environment.json',describe(env))
    write_json(run/'metrics/replay_memory.json',dict(capacity=config.capacity,estimated_bytes=replay.estimated_bytes,
        available_host_bytes=psutil.virtual_memory().available,storage='Both state and next_state int8 CHW; metadata arrays'))
    flags={'stop':False}
    def request_stop(signum,frame):flags['stop']=True;print(f'Signal {signum}: saving full recovery at next episode boundary',flush=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    start=time.monotonic();validation_seconds=0.;reason='episodes';session_updates=agent.updates;session_steps=steps
    def elapsed():return elapsed_base+time.monotonic()-start
    def validation(episode):
        nonlocal best,validation_seconds
        tick=time.monotonic();data=evaluate_policy(agent,list(range(600000,600000+config.eval_episodes)))
        result=summarize(data);validation_seconds+=time.monotonic()-tick
        data.to_csv(run/'metrics'/f'validation_ep{episode:05d}.csv',index=False)
        improved=result['mean']>best
        if improved:
            best=result['mean'];filename=f'best_ep{episode:05d}{suffix}.pt'
            save_model(run/'checkpoints'/filename,agent,episode=episode,steps=steps,frames=frames,validation=result)
            write_json(run/'metrics/selection.json',dict(checkpoint=filename,episode=episode,mean=best,
                protocol='Highest mean on fixed validation seeds 600000..600009; earliest wins ties; held-out test not used'))
        return dict(episode=episode,steps=steps,frames=frames,**result)
    fields=['episode','reward','length','raw_frames','steps','total_raw_frames','epsilon','mean_loss','mean_grad_norm',
        'mean_batch_q','max_abs_batch_q','reference_max_q_mean','reference_max_q_std','optimizer_updates','target_updates',
        'replay_occupancy','trailing100_mean','elapsed_seconds','terminated','truncated','life_losses','zero_reward_steps',
        'max_zero_reward_streak','rss_bytes','gpu_allocated_bytes']
    episode=completed
    try:
        with (run/'logs/episodes.csv').open(mode,newline='') as ef,(run/'logs/updates.csv').open(mode,newline='') as uf,\
             (run/'logs/transitions.csv').open(mode,newline='') as tf,(run/'logs/validation.csv').open(mode,newline='') as vf:
            ew=csv.DictWriter(ef,fieldnames=fields);uw=csv.writer(uf);tw=csv.writer(tf)
            vw=csv.DictWriter(vf,fieldnames=['episode','steps','frames',*summarize_schema()])
            if not resume:
                ew.writeheader();uw.writerow(['update','step','loss','grad_norm','mean_q','max_abs_q','target_updates'])
                tw.writerow(['episode','step','action','reward','terminated','truncated','lives','life_lost','raw_frames','state_adler32','next_state_adler32'])
                vw.writeheader();vw.writerow(validation(0));vf.flush()
            for episode in range(completed+1,config.episodes+1):
                obs,info=env.reset(seed=config.seed+episode-1);state=preprocess_chw(obs)
                score=0.;length=episode_frames=life_losses=zero_count=zero_streak=max_zero_streak=0
                losses=[];norms=[];qs=[];maxq=[];lives=env.unwrapped.ale.lives()
                while True:
                    epsilon=config.epsilon(steps);action=agent.act(state,epsilon,behavior)
                    before=env.unwrapped.ale.getFrameNumber();obs,r,term,trunc,info=env.step(action)
                    delta=env.unwrapped.ale.getFrameNumber()-before
                    frames+=delta;episode_frames+=delta;steps+=1;length+=1;score+=r
                    new_lives=env.unwrapped.ale.lives();life_lost=new_lives<lives;life_losses+=int(life_lost);lives=new_lives
                    zero_count+=int(r==0);zero_streak=zero_streak+1 if r==0 else 0;max_zero_streak=max(max_zero_streak,zero_streak)
                    next_state=preprocess_chw(obs);replay.add(state,action,r,next_state,term,trunc)
                    tw.writerow([episode,steps,action,r,term,trunc,lives,life_lost,delta,zlib.adler32(state),zlib.adler32(next_state)])
                    state=next_state
                    if len(replay)>=config.warmup and steps%config.update_every==0:
                        result=agent.update(replay.sample(config.batch_size,config.replay_mode))
                        losses.append(result['loss']);norms.append(result['grad_norm']);qs.append(result['mean_q']);maxq.append(result['max_abs_q'])
                        uw.writerow([agent.updates,steps,*[result[key] for key in ['loss','grad_norm','mean_q','max_abs_q','target_updates']]])
                    if term or trunc:break
                history.append(score);reference_q=agent.q_values(refs).max(1)
                row=dict(episode=episode,reward=score,length=length,raw_frames=episode_frames,steps=steps,total_raw_frames=frames,
                    epsilon=epsilon,mean_loss=float(np.mean(losses)) if losses else '',mean_grad_norm=float(np.mean(norms)) if norms else '',
                    mean_batch_q=float(np.mean(qs)) if qs else '',max_abs_batch_q=max(maxq) if maxq else '',
                    reference_max_q_mean=float(reference_q.mean()),reference_max_q_std=float(reference_q.std()),
                    optimizer_updates=agent.updates,target_updates=agent.target_updates,replay_occupancy=len(replay),
                    trailing100_mean=float(np.mean(history)) if len(history)==100 else '',elapsed_seconds=elapsed(),
                    terminated=term,truncated=trunc,life_losses=life_losses,zero_reward_steps=zero_count,max_zero_reward_streak=max_zero_streak,
                    rss_bytes=psutil.Process().memory_info().rss,gpu_allocated_bytes=torch.cuda.memory_allocated())
                ew.writerow(row);ef.flush();uf.flush();tf.flush()
                final=steps>=config.max_steps or elapsed()>=config.max_hours*3600 or episode==config.episodes or flags['stop']
                if episode%config.eval_every==0 or final:
                    score_result=validation(episode);vw.writerow(score_result);vf.flush()
                    print(f'episode={episode} steps={steps} frames={frames} train={score:.0f} avg100={np.mean(history):.1f} validation={score_result["mean"]:.1f} best={best:.1f} epsilon={epsilon:.4f} elapsed={elapsed():.1f}s',flush=True)
                elif episode%10==0:
                    print(f'episode={episode} steps={steps} train={score:.0f} avg100={np.mean(history):.1f} epsilon={epsilon:.4f} elapsed={elapsed():.1f}s',flush=True)
                if episode%config.checkpoint_every==0:
                    save_model(run/'checkpoints'/f'stage_ep{episode:05d}{suffix}.pt',agent,episode=episode,steps=steps,frames=frames)
                if episode%config.recovery_every==0 or final:
                    # A small real-transition batch survives independently of large recovery arrays.
                    sample=replay.sample(min(32,len(replay)))
                    for key in ['terminated','truncated']:
                        indices=np.flatnonzero(replay.data[key][:len(replay)])[:4]
                        if len(indices):sample={k:np.concatenate([v,replay.data[k][indices]]) for k,v in sample.items()}
                    np.savez_compressed(run/'metrics/diagnostic_transitions.npz',**sample)
                    save_recovery(run,agent,replay,behavior,episode,steps,frames,elapsed(),best,reference_sha,suffix)
                if final:
                    reason='signal' if flags['stop'] else 'max_steps' if steps>=config.max_steps else 'max_hours' if elapsed()>=config.max_hours*3600 else 'episodes'
                    break
            final_name=f'final_ep{episode:05d}{suffix}.pt'
            save_model(run/'checkpoints'/final_name,agent,episode=episode,steps=steps,frames=frames)
            write_json(run/'metrics/training_summary.json',dict(episodes=episode,steps=steps,raw_frames=frames,
                optimizer_updates=agent.updates,target_updates=agent.target_updates,elapsed_seconds=elapsed(),
                session_seconds=time.monotonic()-start,validation_seconds_this_session=validation_seconds,
                session_steps_per_second=(steps-session_steps)/(time.monotonic()-start),
                session_updates_per_second=(agent.updates-session_updates)/(time.monotonic()-start),
                best_validation_mean=best,stop_reason=reason,epsilon=config.epsilon(steps),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),rss_bytes=psutil.Process().memory_info().rss,
                model_parameters=sum(p.numel() for p in agent.online.parameters()),flattened_conv_size=agent.online.flattened,
                last_update_devices=getattr(agent,'last_update_devices',{}),resume_checkpoint=resume,
                final_checkpoint=final_name,effective_config=asdict(config)))
    except BaseException:
        write_json(run/'metrics'/f'failure_step{steps}.json',dict(episode=episode,steps=steps,traceback=traceback.format_exc(),
            note='Recover from latest completed episode-boundary recovery; in-flight episode not claimed reproducible'))
        raise
    finally:env.close()


def summarize_schema():
    return ['episodes','mean','std_population','median','minimum','maximum','mean_decision_steps','mean_raw_frames','terminated_episodes','truncated_episodes']


def main():
    p=argparse.ArgumentParser();add_args(p);p.add_argument('--output',required=True);p.add_argument('--resume')
    a=p.parse_args();train(from_args(a),a.output,a.resume)


if __name__=='__main__':main()
