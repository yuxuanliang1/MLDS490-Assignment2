"""Final measured-data, artifact, and protected-CartPole integrity audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .runtime import write_json


PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parent


def read(path):return json.loads(Path(path).read_text())


def audit(run):
    run=Path(run).resolve();root=PROJECT/'results/mspacman'
    before=read(root/'cartpole_preservation_before.json')
    normalized={path.replace('\\','/'):digest for path,digest in before.items()}
    changed=[path for path,digest in normalized.items() if not (WORKSPACE/path).exists() or
        hashlib.sha256((WORKSPACE/path).read_bytes()).hexdigest()!=digest]
    if changed:raise AssertionError(f'Protected CartPole files changed: {changed}')
    write_json(root/'cartpole_preservation_after.json',dict(verified_files=len(before),changed=[],passed=True))
    cfg=read(run/'metrics/config.json');training=read(run/'metrics/training_summary.json')
    selection=read(run/'metrics/selection.json');evaluation=read(run/'evaluation500/summary.json')
    episodes=pd.read_csv(run/'logs/episodes.csv');transitions=pd.read_csv(run/'logs/transitions.csv')
    updates=pd.read_csv(run/'logs/updates.csv');test=pd.read_csv(run/'evaluation500/episodes.csv')
    assert cfg['env_id']=='MsPacman-v5' and cfg['gamma']==.99
    assert len(test)==test.seed.nunique()==500 and set(test.seed)==set(range(900000,900500))
    assert (test.terminated|test.truncated).all()
    np.testing.assert_allclose(test.reward.mean(),evaluation['mean'],atol=1e-10)
    np.testing.assert_allclose(test.reward.std(ddof=0),evaluation['std_population'],atol=1e-10)
    assert len(episodes)==training['episodes'] and len(transitions)==training['steps']==episodes.length.sum()
    assert len(updates)==training['optimizer_updates']
    assert transitions.raw_frames.sum()==training['raw_frames']==episodes.raw_frames.sum()
    np.testing.assert_allclose(transitions.groupby('episode').reward.sum(),episodes.reward.to_numpy())
    assert int((transitions.terminated|transitions.truncated).sum())==len(episodes)
    assert (transitions.groupby('episode').tail(1).terminated|transitions.groupby('episode').tail(1).truncated).all()
    assert transitions.action.between(0,8).all()
    assert np.isfinite(updates[['loss','grad_norm','mean_q','max_abs_q']]).all().all()
    assert selection['episode']==evaluation['checkpoint_episode']
    assert (run/'checkpoints'/selection['checkpoint']).exists()
    reference=read(run/'metrics/reference_protocol.json')
    assert hashlib.sha256((run/'metrics/reference_images.npy').read_bytes()).hexdigest()==reference['sha256']
    for name in ['maximum_q_values','training_rewards','evaluation_histogram','state_action_preferences','optimization_diagnostics','learning_stages']:
        for suffix in ['png','pdf']:
            path=run/'figures'/f'{name}.{suffix}'
            assert path.exists() and path.stat().st_size>5000
    diagnostic=read(run/'diagnostics/summary.json')
    assert diagnostic['first_ten_steps']==10
    assert diagnostic['greedy_argmax_matches']==diagnostic['first_ten_steps']+diagnostic['later_window_steps']
    assert diagnostic['terminal_samples']>0
    recovery=read(run/'metrics/latest_recovery.json')
    assert (run/recovery['checkpoint']).exists() and (run/recovery['replay_file']).exists()
    comparison=pd.read_csv(root/'validation/comparison.csv')
    assert len(comparison)==6 and set(comparison.seed)=={41,52,63}
    extended=read(root/'smoke_recovery_extended/summary.json')
    assert extended['passed'] and extended['rollback_parameter_difference']==0
    ledger=root/'pipeline_commands.jsonl'
    statuses=[json.loads(line)['returncode'] for line in ledger.read_text().splitlines() if line]
    assert statuses and all(status==0 for status in statuses)
    assert (PROJECT/'REPORT_NOTES_MSPACMAN.md').exists()
    # Keep uncompressed local analysis logs while providing Git-friendly compressed copies.
    for directory in [run,*[path for path in (root/'validation').iterdir() if path.is_dir()]]:
        source=directory/'logs/transitions.csv'
        if source.exists():
            with source.open('rb') as src,gzip.open(source.with_suffix('.csv.gz'),'wb',compresslevel=6) as target:
                shutil.copyfileobj(src,target)
    paths=[*PROJECT.glob('mspacman/*.py'),PROJECT/'launch_mspacman.py',PROJECT/'tests/test_mspacman.py',
           PROJECT/'requirements_mspacman.txt',PROJECT/'README.md',PROJECT/'REPORT_NOTES_MSPACMAN.md',
           WORKSPACE/'.gitignore',
           run/'checkpoints'/selection['checkpoint'],run/recovery['checkpoint'],run/recovery['replay_file']]
    manifest={str(path.relative_to(WORKSPACE)).replace('\\','/'):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    write_json(root/'source_model_sha256.json',manifest)
    inventory=[str(path.relative_to(WORKSPACE)).replace('\\','/') for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    inventory.extend(str(path.relative_to(WORKSPACE)).replace('\\','/') for path in paths)
    inventory.append('assignment2/FILES_MSPACMAN.md')
    (PROJECT/'FILES_MSPACMAN.md').write_text('# MsPacman files and artifacts\n\n'
        'CartPole source/results are protected separately by the before/after SHA256 audit. '
        'The shared README and .gitignore were extended; no CartPole data were overwritten.\n\n'+
        '\n'.join('- `'+path+'`' for path in sorted(set(inventory)))+'\n',encoding='utf-8')
    result=dict(passed=True,protected_cartpole_files=len(before),main_episodes=len(episodes),main_decisions=len(transitions),
        held_out_episodes=len(test),completed_short_runs=len(comparison),extended_resume_parameter_error=extended['rollback_parameter_difference'],
        source_and_recovery_hashes=len(manifest),required_figures_verified=True,training_accounting_verified=True)
    write_json(root/'artifact_audit.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='assignment2/results/mspacman/main_seed41')
    a=p.parse_args();audit(a.run)


if __name__=='__main__':main()
