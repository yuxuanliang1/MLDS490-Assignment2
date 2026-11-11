"""CUDA smoke and deterministic boundary-resume experiment before main training."""
import argparse
from dataclasses import replace,asdict
from pathlib import Path
import subprocess
import sys
import torch
from .config import Config
from .agent import load_agent
from .runtime import write_json


def run(args):
    command=[sys.executable,'-m','assignment2.mspacman.train',*args]
    result=subprocess.run(command,check=False)
    if result.returncode:raise RuntimeError(f'Smoke failed: {command}')
    return command


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='assignment2/results/mspacman/smoke')
    p.add_argument('--rollback-test',action='store_true')
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=False)
    common=['--capacity','256','--warmup','64','--batch-size','8','--eval-every','2','--eval-episodes','2',
            '--checkpoint-every','2','--recovery-every','2','--max-steps','50000','--epsilon-steps','20000','--reference-count','8']
    if a.rollback_test:common+=['--target-interval','50']
    commands=[]
    commands.append(run(['--output',str(root/'continuous'),'--episodes','4',*common]))
    commands.append(run(['--output',str(root/'split'),'--episodes','2',*common]))
    import json
    recovery=json.loads((root/'split/metrics/latest_recovery.json').read_text())
    commands.append(run(['--output',str(root/'split'),'--episodes','4','--resume',
                         str(root/'split'/recovery['checkpoint']),*common]))
    uninterrupted,_=load_agent(root/'continuous/checkpoints/final_ep00004.pt','cpu')
    resumed,_=load_agent(root/'split/checkpoints/final_ep00004_resume01.pt','cpu')
    differences={key:float((value-resumed.online.state_dict()[key]).abs().max())
                 for key,value in uninterrupted.online.state_dict().items()}
    assert max(differences.values())==0, differences
    continuous=json.loads((root/'continuous/metrics/training_summary.json').read_text())
    split=json.loads((root/'split/metrics/training_summary.json').read_text())
    assert continuous['steps']==split['steps'] and continuous['optimizer_updates']==split['optimizer_updates']
    rollback_difference=None
    if a.rollback_test:
        # Earlier recovery intentionally leaves later log rows/checkpoints present.
        recovery2=sorted((root/'continuous/checkpoints').glob('recovery_ep00002_step*.pt'))[0]
        commands.append(run(['--output',str(root/'continuous'),'--episodes','4',
                             '--resume',str(recovery2),*common]))
        rolled_back,_=load_agent(root/'continuous/checkpoints/final_ep00004_resume01.pt','cpu')
        rollback_difference=max(float((value-rolled_back.online.state_dict()[key]).abs().max())
            for key,value in uninterrupted.online.state_dict().items())
        assert rollback_difference==0
        import pandas as pd
        rows=pd.read_csv(root/'continuous/logs/episodes.csv')
        assert rows.episode.tolist()==[1,2,3,4]
    result=dict(passed=True,commands=commands,max_parameter_difference=max(differences.values()),
        rollback_parameter_difference=rollback_difference,
        continuous=continuous,resumed=split,note='Same initial seed, settings, reference frames, replay and optimizer/RNG restored at episode boundary')
    write_json(root/'summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
