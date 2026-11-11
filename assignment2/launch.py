"""Detached sequential experiment driver; prints and saves only its own PID."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('phase',choices=['main','validation','primary'])
a=p.parse_args()
root=Path(__file__).resolve().parent.parent
log=root/f'{a.phase}_driver.log'
if log.exists():
    raise SystemExit(f'Refusing to overwrite {log}')
with log.open('w') as output:
    process=subprocess.Popen(['bash',f'assignment2/run_{a.phase}.sh'],cwd=root,
        stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(root/f'{a.phase}_driver.pid').write_text(str(process.pid)+'\n')
print(f'pid={process.pid} log={log}',flush=True)
