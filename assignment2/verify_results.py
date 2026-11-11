"""Final artifact consistency audit, independent of training execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results/cartpole'
RUNS=['main_seed11','seed22','seed33','soft_target_seed11','target_every_update_seed11',
      'recent_replay_seed11','gamma090_seed11','gamma099_seed11']


def read(path):return json.loads(path.read_text())


def main():
    checked=[]
    for name in RUNS:
        run=RESULTS/name
        config=read(run/'metrics/config.json')
        audit=read(run/'metrics/gpu_audit.json')
        training=read(run/'metrics/training_summary.json')
        choice=read(run/'metrics/selection.json')
        summary=read(run/'evaluation500/summary.json')
        evaluation=pd.read_csv(run/'evaluation500/episodes.csv')
        episodes=pd.read_csv(run/'logs/episodes.csv')
        updates=pd.read_csv(run/'logs/updates.csv')
        transitions=pd.read_csv(run/'logs/transitions.csv')
        assert config['env_id']=='CartPole-v1'
        if name not in ['gamma090_seed11','gamma099_seed11']:assert config['gamma']==.95
        assert audit['hostname']=='mlds-deepdish5' and audit['cuda_matmul_sum']==3680.
        assert audit['cuda_visible_devices']=='GPU-cef65388-88b0-f39a-6bac-c5316eb78540'
        assert training['model_device']==training['gradient_device']=='cuda:0'
        assert len(evaluation)==500 and evaluation.seed.nunique()==500
        assert set(evaluation.seed)==set(range(100000,100500))
        np.testing.assert_allclose(evaluation.reward.mean(),summary['mean'],atol=1e-10)
        np.testing.assert_allclose(evaluation.reward.std(ddof=0),summary['std_population'],atol=1e-10)
        assert int((evaluation.reward==500).sum())==summary['perfect_episodes']
        assert (evaluation.reward==evaluation.length).all()
        assert (evaluation.terminated|evaluation.truncated).all()
        assert (evaluation.length<=500).all() and (evaluation.length>=1).all()
        assert len(episodes)==training['episodes']
        assert len(transitions)==training['env_steps']==episodes.length.sum()
        assert len(updates)==training['optimizer_updates']
        assert (episodes.reward==episodes.length).all()
        np.testing.assert_allclose(transitions.groupby('episode').reward.sum(),episodes.reward)
        # Exactly one end transition per episode, always its final row.
        assert int((transitions.terminated|transitions.truncated).sum())==len(episodes)
        final_rows=transitions.groupby('episode').tail(1)
        assert (final_rows.terminated|final_rows.truncated).all()
        assert np.isfinite(updates[['loss','grad_norm','max_abs_q']]).all().all()
        assert (run/'checkpoints'/choice['checkpoint']).exists()
        assert (run/'checkpoints/final.pt').exists()
        assert summary['checkpoint_episode']==choice['episode']
        for figure in ['initial_state_max_q','training_rewards','evaluation_histogram','policy_action_map','optimization_diagnostics']:
            for suffix in ['png','pdf']:
                path=run/'figures'/f'{figure}.{suffix}'
                assert path.exists() and path.stat().st_size>5000
        grid=pd.read_csv(run/'metrics/policy_grid.csv')
        assert len(grid)==161*161
        assert (grid.action==grid[['q_left','q_right']].to_numpy().argmax(1)).all()
        checked.append(dict(run=name,episodes=len(episodes),steps=len(transitions),test_episodes=len(evaluation),
                            successful_data_consistency=True))
    assert (ROOT.parent/'validation_exit_status.txt').read_text().strip()=='0'
    selection=read(RESULTS/'primary_selection.json')
    primary=RESULTS/selection['run']
    diagnostic=read(primary/'diagnostics/summary.json')
    assert diagnostic['trajectory_steps']==diagnostic['greedy_action_matches']==10
    assert diagnostic['max_bellman_absolute_error']<2e-5
    stress=pd.read_csv(primary/'robustness/episodes.csv')
    assert len(stress)==405
    assert (ROOT/'REPORT_NOTES_CARTPOLE.md').exists()
    output=RESULTS/'artifact_verification.json'
    output.write_text(json.dumps(dict(passed=True,runs=checked,primary=selection['run'],
        primary_test_episodes=500,primary_stress_episodes=405),indent=2)+'\n')
    delivered=[str(p.relative_to(ROOT.parent)).replace('\\','/') for p in ROOT.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts]
    delivered += [p.name for p in ROOT.parent.iterdir() if p.is_file() and
                  (p.name=='.gitignore' or p.name=='gpu_audit.json' or p.suffix=='.log' or
                   p.name.endswith('_status.txt') or p.name.endswith('_driver.pid'))]
    delivered=sorted(set(delivered+['assignment2/FILES_CREATED.md']))
    (ROOT/'FILES_CREATED.md').write_text('# Created project and result files\n\n'
        'Reference PDFs were supplied by the user and were not modified. Ignored temporary '
        'extractor packages, caches and transfer archives are omitted.\n\n'+
        '\n'.join('- `'+path+'`' for path in delivered)+'\n',encoding='utf-8')
    # Hash source, report and each selected checkpoint to make the delivered version explicit.
    manifest={}
    paths=[*ROOT.glob('*.py'),*ROOT.glob('*.md'),*ROOT.glob('*.sh'),*ROOT.glob('cartpole/*.py'),
           *ROOT.glob('tests/*.py'),ROOT/'requirements.txt']
    paths += [RESULTS/name/'checkpoints'/read(RESULTS/name/'metrics/selection.json')['checkpoint'] for name in RUNS]
    for path in sorted(paths):
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULTS/'source_checkpoint_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'PASS: {len(checked)} training runs; 4000 held-out episodes; all required figures, flags and accounting verified')


if __name__=='__main__':main()
