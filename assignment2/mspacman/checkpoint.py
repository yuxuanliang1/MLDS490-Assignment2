"""Episode-boundary recovery with replay, optimizer, and independent RNG state."""
from dataclasses import asdict
import os
from pathlib import Path
import csv
import json
import shutil
import torch
from .agent import cpu_tree,DQN
from .config import Config
from .replay_buffer import ReplayBuffer
from .runtime import rng_state,restore_rng,write_json


def save_recovery(run,agent,replay,behavior_rng,episode,steps,frames,elapsed,best,reference_sha,session_suffix=''):
    run=Path(run);name=f'recovery_ep{episode:05d}_step{steps:09d}{session_suffix}'
    replay_path=run/'replay'/f'{name}.npz';model_path=run/'checkpoints'/f'{name}.pt'
    if replay_path.exists() or model_path.exists():raise FileExistsError(name)
    replay.save(replay_path)
    data=dict(kind='full_episode_boundary',config=asdict(agent.config),online=cpu_tree(agent.online.state_dict()),
        target=cpu_tree(agent.target.state_dict()),optimizer=cpu_tree(agent.optimizer.state_dict()),
        updates=agent.updates,target_updates=agent.target_updates,episode=episode,steps=steps,frames=frames,
        elapsed_seconds=elapsed,best_validation=best,replay_file=str(replay_path.relative_to(run)),
        behavior_rng=behavior_rng.bit_generator.state,rng=rng_state(),reference_sha256=reference_sha,
        selection=json.loads((run/'metrics/selection.json').read_text()),
        epsilon=agent.config.epsilon(steps),resume_protocol='Boundary reset seed = training seed + next episode index - 1')
    temporary=model_path.with_suffix('.tmp');torch.save(data,temporary);os.replace(temporary,model_path)
    write_json(run/'metrics/latest_recovery.json',dict(checkpoint=str(model_path.relative_to(run)),
        replay_file=data['replay_file'],episode=episode,steps=steps,replay_bytes=replay_path.stat().st_size))
    return model_path


def load_recovery(path,run,device):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['kind']!='full_episode_boundary':raise ValueError('Model-only checkpoints are not equivalent training resumes')
    agent=DQN(Config(**data['config']),device)
    agent.online.load_state_dict(data['online']);agent.target.load_state_dict(data['target'])
    agent.optimizer.load_state_dict(data['optimizer']);agent.updates=data['updates'];agent.target_updates=data['target_updates']
    replay=ReplayBuffer.load(Path(run)/data['replay_file'])
    restore_rng(data['rng'])
    return agent,replay,data


def rollback_logs(run,data,session):
    """Preserve then remove log rows beyond a recovered boundary, never duplicate them."""
    run=Path(run);archive=run/'logs'/f'archive_before_resume_{session:02d}';archive.mkdir(exist_ok=False)
    thresholds={'episodes.csv':('episode',data['episode']),'updates.csv':('step',data['steps']),
                'transitions.csv':('step',data['steps']),'validation.csv':('episode',data['episode'])}
    for filename,(key,limit) in thresholds.items():
        path=run/'logs'/filename
        shutil.copy2(path,archive/filename)
        temporary=path.with_suffix('.rollback')
        with path.open(newline='') as source,temporary.open('w',newline='') as target:
            reader=csv.DictReader(source);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
            for row in reader:
                try:include=int(row[key])<=limit
                except (ValueError,TypeError):continue # Incomplete final row after abrupt interruption.
                if include:writer.writerow(row)
        os.replace(temporary,path)
    for path in (run/'metrics').glob('validation_ep*.csv'):
        episode=int(path.stem.split('ep')[-1])
        if episode>data['episode']:shutil.copy2(path,archive/path.name)
    for filename in ['selection.json','training_summary.json']:
        path=run/'metrics'/filename
        if path.exists():shutil.copy2(path,archive/filename)
    if 'selection' in data:
        selection=data['selection']
    else:
        # Backward-compatible with the first smoke/main recovery schema.
        candidates=[]
        for path in (run/'checkpoints').glob('best_ep*.pt'):
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['episode']<=data['episode']:
                candidates.append((checkpoint['validation']['mean'],-checkpoint['episode'],path.name))
        mean,negative_episode,name=max(candidates)
        if mean!=data['best_validation']:raise RuntimeError('Cannot reconcile checkpoint validation selection with recovery')
        selection=dict(checkpoint=name,episode=-negative_episode,mean=mean,
            protocol='Recovered highest validation mean at or before restored episode boundary')
    write_json(run/'metrics/selection.json',selection)
    return archive
