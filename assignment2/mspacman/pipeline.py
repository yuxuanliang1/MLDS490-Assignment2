"""Main experiment; final evaluation starts only after training ends successfully."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='assignment2/results/mspacman/main_seed41')
    p.add_argument('--episodes',type=int,default=4000);p.add_argument('--max-steps',type=int,default=2000000)
    p.add_argument('--max-hours',type=float,default=4.)
    a=p.parse_args()
    smoke=json.loads(Path('assignment2/results/mspacman/smoke/summary.json').read_text())
    if not smoke['passed']:raise RuntimeError('GPU smoke/recovery experiment must pass before main training')
    commands=[['-m','assignment2.mspacman.train','--output',a.output,'--episodes',str(a.episodes),
               '--max-steps',str(a.max_steps),'--max-hours',str(a.max_hours)],
              ['-m','assignment2.mspacman.validate','finalize','--run',a.output]]
    ledger=Path('assignment2/results/mspacman/pipeline_commands.jsonl')
    for command in commands:
        start=time.time();result=subprocess.run([sys.executable,*command],check=False)
        with ledger.open('a') as f:f.write(json.dumps(dict(command=[sys.executable,*command],returncode=result.returncode,
            started_unix=start,elapsed_seconds=time.time()-start))+'\n')
        if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
