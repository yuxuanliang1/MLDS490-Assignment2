"""Detached single-job execution with durable command, PID and exit-status evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parent.parent
JOBS=ROOT/'assignment2/results/mspacman/jobs'


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--after',help='Queue a cheap waiting driver behind a named successful job')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();JOBS.mkdir(parents=True,exist_ok=True)
    metadata=JOBS/f'{args.label}.json';log=JOBS/f'{args.label}.log';status=JOBS/f'{args.label}_exit.json'
    if args.worker:
        job=json.loads(metadata.read_text());start=time.time()
        try:
            if job.get('after'):
                dependency=JOBS/f'{job["after"]}_exit.json'
                print(f'Queued behind {job["after"]}; no training/evaluation runs concurrently',flush=True)
                while not dependency.exists():
                    other=json.loads((JOBS/f'{job["after"]}.json').read_text())
                    try:os.kill(other['pid'],0)
                    except ProcessLookupError:raise RuntimeError('Dependency stopped without an exit-status file')
                    time.sleep(5)
                if json.loads(dependency.read_text())['returncode']!=0:
                    raise RuntimeError(f'Dependency {job["after"]} failed; queued experiment will not start')
            queued=time.time()-start;start=time.time()
            result=subprocess.run(job['command'],cwd=ROOT,check=False)
            write(status,dict(returncode=result.returncode,started_unix=start,queued_seconds=queued,elapsed_seconds=time.time()-start))
        except BaseException as error:
            write(status,dict(returncode=-1,error=repr(error),started_unix=start));raise
        raise SystemExit(result.returncode)
    if metadata.exists():raise SystemExit(f'Use a new job label; {metadata} already exists')
    for existing in JOBS.glob('*.json'):
        if existing.name.endswith('_exit.json'):continue
        other=json.loads(existing.read_text())
        if (JOBS/f'{existing.stem}_exit.json').exists():continue
        try:os.kill(other['pid'],0)
        except (ProcessLookupError,KeyError):continue
        if args.after==existing.stem:continue
        raise SystemExit(f'Another experiment driver is active: {existing}')
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise SystemExit('Provide a command after the label')
    if args.after and not (JOBS/f'{args.after}.json').exists():raise SystemExit('Unknown dependency job')
    write(metadata,dict(command=command,created_unix=time.time(),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),after=args.after))
    with log.open('x') as output:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',args.label],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    value=json.loads(metadata.read_text());value['pid']=process.pid;write(metadata,value)
    print(f'pid={process.pid} log={log}',flush=True)


if __name__=='__main__':main()
