"""Measured learning, Atari edge cases, Bellman references, and bounded ablations."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from .agent import load_agent,bellman_target
from .environment import make_env
from .evaluate import evaluate_policy,summarize
from .preprocessing import preprocess_chw
from .runtime import gpu_audit,write_json


def selected(run):
    run=Path(run);choice=json.loads((run/'metrics/selection.json').read_text())
    return run/'checkpoints'/choice['checkpoint']


def execute(args,ledger):
    start=time.time();command=[sys.executable,*args];result=subprocess.run(command,check=False)
    with Path(ledger).open('a') as f:f.write(json.dumps(dict(command=command,returncode=result.returncode,
        started_unix=start,elapsed_seconds=time.time()-start))+'\n')
    if result.returncode:raise RuntimeError(f'Failed command: {command}')


def random_control(seeds):
    rows=[]
    for seed in seeds:
        env=make_env(seed);obs,_=env.reset(seed=seed);rng=np.random.default_rng(seed+1234567)
        score=0.;steps=frames=0;lost=0;lives=env.unwrapped.ale.lives()
        try:
            while True:
                before=env.unwrapped.ale.getFrameNumber()
                obs,reward,term,trunc,info=env.step(int(rng.integers(9)))
                frames+=env.unwrapped.ale.getFrameNumber()-before;score+=reward;steps+=1
                current=env.unwrapped.ale.lives();lost+=int(current<lives);lives=current
                if term or trunc:break
        finally:env.close()
        rows.append(dict(seed=seed,reward=score,length=steps,raw_frames=frames,terminated=term,truncated=trunc,life_losses=lost))
    return pd.DataFrame(rows)


def learning_stages(run):
    run=Path(run);validation=pd.read_csv(run/'logs/validation.csv')
    choice=json.loads((run/'metrics/selection.json').read_text())
    targets=[('initial',0),('early',int(validation.episode.iloc[min(1,len(validation)-1)])),
             ('intermediate',int(validation.iloc[(validation.steps-validation.steps.max()/2).abs().argmin()].episode)),
             ('selected',choice['episode'])]
    rows=[]
    for label,episode in targets:
        record=validation.loc[validation.episode==episode].iloc[0].to_dict();record['stage']=label;rows.append(record)
    table=pd.DataFrame(rows);table.to_csv(run/'metrics/learning_stages.csv',index=False)
    initial=pd.read_csv(run/'metrics/validation_ep00000.csv').set_index('seed').reward
    final=pd.read_csv(run/'metrics'/f'validation_ep{choice["episode"]:05d}.csv').set_index('seed').reward
    differences=(final-initial).to_numpy();rng=np.random.default_rng(7788)
    boot=differences[rng.integers(len(differences),size=(10000,len(differences)))].mean(1)
    control=random_control(list(range(600000,600000+len(initial))))
    control.to_csv(run/'metrics/random_policy_validation.csv',index=False)
    write_json(run/'metrics/learning_evidence.json',dict(initial_validation_mean=float(initial.mean()),
        selected_validation_mean=float(final.mean()),paired_validation_improvement=float(differences.mean()),
        paired_bootstrap_95_percent_interval=np.quantile(boot,[.025,.975]),selection_panel_episodes=len(initial),
        interval_limitation='Exploratory paired bootstrap on reused selection panel; not an unbiased confirmatory interval',
        random_control=summarize(control)))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plotting import save
    fig,ax=plt.subplots(figsize=(9,5))
    ax.errorbar(table.stage,table['mean'],yerr=table.std_population,fmt='o-',color='#236e9a',capsize=5,label='Mean ± episode population SD')
    ax.axhline(control.reward.mean(),ls='--',color='#b45121',label='Uniform-random policy mean on same reset seeds')
    ax.set(xlabel='Training stage',ylabel='Undiscounted validation score',title='Evidence of learning under a common validation protocol');ax.legend(fontsize=9)
    save(fig,run/'figures','learning_stages','Fixed validation seeds; error bars show within-policy episode variability, not confidence intervals. Selection bias is acknowledged.')


def diagnostics(run):
    run=Path(run);output=run/'diagnostics';output.mkdir(exist_ok=False)
    write_json(output/'gpu_audit.json',gpu_audit());agent,_=load_agent(selected(run),'cuda:0')
    with np.load(run/'metrics/diagnostic_transitions.npz',allow_pickle=False) as archive:
        batch={key:archive[key] for key in archive.files}
    next_tensor=torch.as_tensor(batch['next_state'],device='cuda:0')
    with torch.no_grad():next_q=agent.target(next_tensor).cpu().numpy()
    reference=np.array([float(r)+.99*(not bool(d))*float(max(values))
        for r,d,values in zip(batch['reward'],batch['terminated'],next_q)],dtype=np.float64)
    actual=bellman_target(agent.target,next_tensor,torch.as_tensor(batch['reward'],device='cuda:0'),
                         torch.as_tensor(batch['terminated'],device='cuda:0'),.99).cpu().numpy()
    errors=np.abs(reference-actual)
    np.testing.assert_allclose(reference,actual,atol=1e-3,rtol=2e-6)
    pd.DataFrame(dict(action=batch['action'],reward=batch['reward'],terminated=batch['terminated'],truncated=batch['truncated'],
        max_target_next_q=next_q.max(1),numpy_reference=reference,torch_target=actual,absolute_error=errors)).to_csv(output/'bellman_reference.csv',index=False)
    np.savez_compressed(output/'sampled_actual_transitions.npz',**batch)
    env=make_env(314159);obs,_=env.reset(seed=314159);state=preprocess_chw(obs)
    rows=[];states=[];next_states=[];rgb=[]
    try:
        for step in range(110):
            q=agent.q_values(state)[0];action=int(q.argmax())
            next_obs,r,term,trunc,info=env.step(action);nxt=preprocess_chw(next_obs)
            if step<10 or step>=100:
                with torch.no_grad():target_next=agent.target(torch.as_tensor(nxt[None],device='cuda:0')).max().item()
                rows.append(dict(step=step+1,**{f'q_{i}':float(q[i]) for i in range(9)},action=action,reward=r,
                    terminated=term,truncated=trunc,lives=env.unwrapped.ale.lives(),target_reference=r+.99*(not term)*target_next,
                    image_index=len(states)))
                states.append(state);next_states.append(nxt);rgb.append(obs)
            obs=next_obs;state=nxt
            if term or trunc:break
    finally:env.close()
    trace=pd.DataFrame(rows);trace.to_csv(output/'trajectory.csv',index=False)
    np.savez_compressed(output/'trajectory_images.npz',state=np.stack(states),next_state=np.stack(next_states),rgb=np.stack(rgb))
    # A portable trained-checkpoint reload on exactly the same images.
    cpu_agent,_=load_agent(selected(run),'cpu');probe=np.stack(states)
    cuda_q=agent.q_values(probe);cpu_q=cpu_agent.q_values(probe)
    # Cross-device numerical equality is measured; near-tied actions can differ.
    write_json(output/'summary.json',dict(actual_transition_count=len(reference),terminal_samples=int(batch['terminated'].sum()),
        truncated_samples=int(batch['truncated'].sum()),max_bellman_absolute_error=float(errors.max()),
        trajectory_seed=314159,first_ten_steps=int((trace.step<=10).sum()),later_window_steps=int((trace.step>=101).sum()),
        greedy_argmax_matches=int((trace.action.to_numpy()==trace[[f'q_{i}' for i in range(9)]].to_numpy().argmax(1)).sum()),
        cpu_cuda_q_max_absolute_difference=float(np.abs(cpu_q-cuda_q).max()),
        cpu_cuda_greedy_agreement=float((cpu_q.argmax(1)==cuda_q.argmax(1)).mean()),
        reference_note='Independent float64 NumPy scalar Bellman arithmetic on saved target-network values; not a claim to reconstruct historical update targets'))
    transitions=pd.read_csv(run/'logs/transitions.csv');episodes=pd.read_csv(run/'logs/episodes.csv')
    unusual=np.stack([preprocess_chw(np.zeros((210,160,3),np.uint8)),preprocess_chw(np.full((210,160,3),255,np.uint8))])
    unusual_q=agent.q_values(unusual)
    write_json(output/'edge_cases.json',dict(game_over_transitions=int(transitions.terminated.sum()),
        truncated_transitions=int(transitions.truncated.sum()),life_loss_nonterminal_transitions=int((transitions.life_lost&~transitions.terminated).sum()),
        unchanged_processed_frame_fraction=float((transitions.state_adler32==transitions.next_state_adler32).mean()),
        zero_reward_transition_fraction=float((transitions.reward==0).mean()),max_zero_reward_streak=int(episodes.max_zero_reward_streak.max()),
        minimum_transition_reward=float(transitions.reward.min()),maximum_transition_reward=float(transitions.reward.max()),
        action_counts=transitions.action.value_counts().sort_index().to_dict(),unusual_black_white_q_values=unusual_q,
        actual_reward_values=transitions.groupby('reward').size().to_dict(),
        termination_note='Loss of a life is not treated as terminal; genuine ALE game-over alone masks bootstrap. Native frame limit is preserved.'))


def finalize(run):
    run=Path(run);ledger=run/'logs/finalization_commands.jsonl'
    execute(['-m','assignment2.mspacman.evaluate','--checkpoint',str(selected(run)),
             '--output',str(run/'evaluation500')],ledger)
    execute(['-m','assignment2.mspacman.plotting','--run',str(run),'--device','cuda:0'],ledger)
    learning_stages(run);diagnostics(run)


def short_suite(root,steps):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'gpu_audit.json',gpu_audit());ledger=root/'commands.jsonl'
    specs=[('baseline_seed41',41,[]),('seed52',52,[]),('seed63',63,[]),
        ('target_every_update_seed41',41,['--target-interval','1']),
        ('recent_replay_seed41',41,['--replay-mode','recent']),
        ('epsilon_slow_seed41',41,['--epsilon-steps','1000000'])]
    for name,seed,changes in specs:
        run=root/name
        execute(['-m','assignment2.mspacman.train','--output',str(run),'--seed',str(seed),
            '--episodes','4000','--max-steps',str(steps),*changes],ledger)
        agent,_=load_agent(selected(run),'cuda:0');data=evaluate_policy(agent,list(range(700000,700030)))
        data.to_csv(run/'metrics/diagnostic_evaluation30.csv',index=False)
        write_json(run/'metrics/diagnostic_evaluation30.json',summarize(data))
    compare(root)


def compare(root):
    root=Path(root);rows=[]
    for run in sorted(root.iterdir()):
        if not (run/'metrics/diagnostic_evaluation30.json').exists():continue
        cfg=json.loads((run/'metrics/config.json').read_text());score=json.loads((run/'metrics/diagnostic_evaluation30.json').read_text())
        train=json.loads((run/'metrics/training_summary.json').read_text());eps=pd.read_csv(run/'logs/episodes.csv')
        updates=pd.read_csv(run/'logs/updates.csv');validation=pd.read_csv(run/'logs/validation.csv')
        rows.append(dict(run=run.name,seed=cfg['seed'],target_interval=cfg['target_interval'],replay_mode=cfg['replay_mode'],epsilon_steps=cfg['epsilon_steps'],
            episodes=train['episodes'],steps=train['steps'],updates=train['optimizer_updates'],elapsed_seconds=train['elapsed_seconds'],
            evaluation_mean=score['mean'],evaluation_std=score['std_population'],initial_validation_mean=validation['mean'].iloc[0],
            best_validation_mean=train['best_validation_mean'],first100_training_mean=float(eps.reward.head(100).mean()),
            last100_training_mean=float(eps.reward.tail(100).mean()),final_reference_q=float(eps.reference_max_q_mean.iloc[-1]),
            loss_mean=float(updates.loss.mean()),loss_p95=float(updates.loss.quantile(.95)),max_abs_q=float(updates.max_abs_q.max()),
            final_epsilon=train['epsilon']))
    table=pd.DataFrame(rows);table.to_csv(root/'comparison.csv',index=False)
    baseline_path=root/'baseline_seed41/metrics/diagnostic_evaluation30.csv'
    if baseline_path.exists():
        baseline=pd.read_csv(baseline_path).set_index('seed').reward
        paired=[];rng=np.random.default_rng(1232026)
        for name in ['target_every_update_seed41','recent_replay_seed41','epsilon_slow_seed41']:
            path=root/name/'metrics/diagnostic_evaluation30.csv'
            if not path.exists():continue
            variant=pd.read_csv(path).set_index('seed').reward
            assert set(variant.index)==set(baseline.index)
            differences=(variant-baseline).to_numpy()
            bootstrap=differences[rng.integers(len(differences),size=(10000,len(differences)))].mean(1)
            low,high=np.quantile(bootstrap,[.025,.975])
            paired.append(dict(variant=name,episodes=len(differences),mean_difference=float(differences.mean()),
                bootstrap95_low=float(low),bootstrap95_high=float(high),resamples=10000))
        pd.DataFrame(paired).to_csv(root/'paired_comparisons.csv',index=False)
    replicated=table[table.run.isin(['baseline_seed41','seed52','seed63'])]
    if len(replicated):write_json(root/'replication_summary.json',dict(runs=len(replicated),
        mean_of_run_means=float(replicated.evaluation_mean.mean()),population_sd_of_run_means=float(replicated.evaluation_mean.std(ddof=0)),
        limitation='Short-budget replications, not independent full-budget main runs'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plotting import save
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    panels=[('Short-budget training seeds',['baseline_seed41','seed52','seed63']),
            ('Target-network lag',['baseline_seed41','target_every_update_seed41']),
            ('Replay sampling',['baseline_seed41','recent_replay_seed41']),
            ('Exploration decay',['baseline_seed41','epsilon_slow_seed41'])]
    for ax,(title,names) in zip(axes.flat,panels):
        for name in names:
            path=root/name/'logs/validation.csv'
            if path.exists():
                data=pd.read_csv(path);ax.plot(data.steps,data['mean'],'o-',ms=3,label=name,linewidth=1)
        ax.set(title=title,xlabel='Agent decision steps',ylabel='Greedy validation mean score');ax.legend(fontsize=7)
    save(fig,root,'short_validation_curves','Matched decision-step caps; one changed factor at seed 41. These short experiments do not establish full-budget reproducibility.')


def same_seed_prefix(main,short):
    main=Path(main);short=Path(short);count=0;difference=None
    with (main/'logs/transitions.csv').open() as left,(short/'logs/transitions.csv').open() as right:
        for a,b in zip(csv.DictReader(left),csv.DictReader(right)):
            count+=1
            if a!=b:difference=dict(step=a['step'],fields=[key for key in a if a[key]!=b[key]]);break
    summary=json.loads((short/'metrics/training_summary.json').read_text())
    common=sorted(set(path.name for path in (main/'checkpoints').glob('stage_ep*.pt'))&
                  set(path.name for path in (short/'checkpoints').glob('stage_ep*.pt')))
    if not common:raise ValueError('No shared saved checkpoint for parameter comparison')
    filename=common[-1]
    a=torch.load(main/'checkpoints'/filename,map_location='cpu',weights_only=True)
    b=torch.load(short/'checkpoints'/filename,map_location='cpu',weights_only=True)
    errors={network:max(float((a[network][key]-b[network][key]).abs().max()) for key in a[network]) for network in ['online','target']}
    assert difference is None and count==summary['steps'] and max(errors.values())==0
    result=dict(matched_seed=41,compared_decisions=count,first_recorded_transition_difference=difference,
        parameter_comparison_episode=a['episode'],max_parameter_differences=errors,
        scope='Same GPU/software: all scalar records and image checksums in short-run prefix; exact online/target parameters at last common saved stage')
    write_json(main.parent/'same_seed_prefix_reproducibility.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['finalize','diagnostics','learning-stages']:
        parser=sub.add_parser(name);parser.add_argument('--run',required=True)
    for name in ['short-suite','compare']:
        parser=sub.add_parser(name);parser.add_argument('--root',default='assignment2/results/mspacman/validation')
        if name=='short-suite':parser.add_argument('--steps',type=int,default=100000)
    parser=sub.add_parser('same-seed-prefix');parser.add_argument('--main',default='assignment2/results/mspacman/main_seed41')
    parser.add_argument('--short',default='assignment2/results/mspacman/validation/baseline_seed41')
    a=p.parse_args()
    if a.command=='finalize':finalize(a.run)
    elif a.command=='diagnostics':diagnostics(a.run)
    elif a.command=='learning-stages':learning_stages(a.run)
    elif a.command=='short-suite':short_suite(a.root,a.steps)
    elif a.command=='same-seed-prefix':same_seed_prefix(a.main,a.short)
    else:compare(a.root)


if __name__=='__main__':main()
