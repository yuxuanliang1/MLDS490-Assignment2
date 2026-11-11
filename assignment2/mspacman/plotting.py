import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .agent import load_agent
from .environment import make_env
from .runtime import write_json


plt.rcParams.update({'font.size':11,'figure.dpi':130,'savefig.dpi':220,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})


def save(fig,out,name,note):
    fig.text(.5,.012,note,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/(name+'.png'));fig.savefig(out/(name+'.pdf'));plt.close(fig)


def draw_preferences(raw,q,names,out):
    indices=np.linspace(0,len(q)-1,6,dtype=int)
    fig,axes=plt.subplots(6,2,figsize=(12,14),gridspec_kw={'width_ratios':[1,2.5]})
    for row,i in enumerate(indices):
        image_ax,value_ax=axes[row];action=int(q[i].argmax())
        image_ax.imshow(raw[i]);image_ax.set_title(f'Reference {i}: {names[action]}',fontsize=10);image_ax.axis('off')
        value_ax.scatter(np.arange(9),q[i],s=45,
            color=['#b45121' if a==action else '#236e9a' for a in range(9)],zorder=3)
        for a,value in enumerate(q[i]):
            value_ax.annotate(f'{value:.2f}',(a,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        value_ax.margins(y=.3);value_ax.set(xlim=(-.5,8.5),ylabel='Q(s,a)')
        value_ax.set_xticks(np.arange(9))
        if row==len(indices)-1:
            value_ax.set_xticklabels(names,rotation=25,ha='right',fontsize=8)
            value_ax.set_xlabel('Commanded action')
        else:value_ax.set_xticklabels([])
    fig.suptitle('MsPacman-v5: state-dependent action values',fontsize=15)
    save(fig,out,'state_action_preferences',
        'Full RGB shown; CNN uses crop/mask. Q is in discounted raw-score units; each row has its own scale. Orange marks greedy action.')


def make_plots(run,device):
    run=Path(run);out=run/'figures';out.mkdir(exist_ok=True)
    data=pd.read_csv(run/'logs/episodes.csv');validation=pd.read_csv(run/'logs/validation.csv')
    selected=json.loads((run/'metrics/selection.json').read_text())
    result=pd.read_csv(run/'evaluation500/episodes.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(data.episode,data.reference_max_q_mean,color='#236e9a',label='Mean of max-action Q on fixed references')
    ax.fill_between(data.episode,(data.reference_max_q_mean-data.reference_max_q_std),
        (data.reference_max_q_mean+data.reference_max_q_std),alpha=.2,color='#236e9a',label='±1 SD across reference images')
    ax.set(xlabel='Completed training episode',ylabel='Predicted discounted raw-score return',title='MsPacman-v5: action values on a fixed image panel')
    ax.legend(fontsize=9)
    save(fig,out,'maximum_q_values','Same early gameplay images at every episode; mean over images of max over 9 actions. Larger Q need not mean better play.')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(data.episode,data.reward,lw=.6,alpha=.25,color='#236e9a',label='Exploratory training score')
    ax.plot(data.episode,data.reward.rolling(100,min_periods=100).mean(),lw=2,color='#163c54',label='Trailing 100-episode mean')
    ax.plot(validation.episode,validation['mean'],'o-',color='#b45121',ms=3,lw=1,label='Separate greedy validation mean')
    ax.set(xlabel='Completed training episode',ylabel='Undiscounted raw game score',title='MsPacman-v5: complete training reward trajectory')
    ax.legend(fontsize=9)
    save(fig,out,'training_rewards','No reward clipping or episodic-life resets. Moving average first appears at episode 100; validation seeds are separate.')
    fig,ax=plt.subplots(figsize=(9,5));rewards=result.reward
    ax.hist(rewards,bins=30,color='#236e9a',edgecolor='white')
    ax.axvline(rewards.mean(),color='#b45121',lw=2,label=f'Mean {rewards.mean():.3f}; population SD {rewards.std(ddof=0):.3f}')
    ax.set(xlabel='Greedy undiscounted raw game score',ylabel='Number of episodes',title='MsPacman-v5: exactly 500 held-out greedy episodes')
    ax.legend(fontsize=10)
    save(fig,out,'evaluation_histogram','Distinct reset seeds 900000–900499; ε=0. Scores include every life and the final transition.')
    agent,_=load_agent(run/'checkpoints'/selected['checkpoint'],device)
    refs=np.load(run/'metrics/reference_images.npy');raw=np.load(run/'metrics/reference_rgb.npy')
    q=agent.q_values(refs)
    initial,_=load_agent(run/'checkpoints/initial.pt',device)
    initial_q=initial.q_values(refs)
    ordered=np.sort(q,axis=1)
    write_json(run/'metrics/action_preference_summary.json',dict(reference_images=len(refs),
        distinct_selected_greedy_actions=int(len(np.unique(q.argmax(1)))),
        fraction_actions_changed_from_initial=float((q.argmax(1)!=initial_q.argmax(1)).mean()),
        median_top_two_q_margin=float(np.median(ordered[:,-1]-ordered[:,-2])),
        mean_within_image_q_range=float(np.mean(q.max(1)-q.min(1)))))
    env=make_env();names=env.unwrapped.get_action_meanings();env.close()
    pd.DataFrame([dict(reference=i,action=a,action_name=names[a],q_value=float(q[i,a]),greedy=a==int(q[i].argmax()))
        for i in range(len(q)) for a in range(9)]).to_csv(run/'metrics/action_preferences.csv',index=False)
    draw_preferences(raw,q,names,out)
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    axes[0].plot(data.steps,data.mean_loss,color='#236e9a');axes[0].set(ylabel='Episode mean Huber loss',title='MsPacman optimization and exploration diagnostics')
    axes[1].plot(data.steps,data.epsilon,color='#b45121');axes[1].set(ylabel='Training ε')
    axes[2].plot(data.steps,data.mean_grad_norm,color='#236e9a');axes[2].axhline(10,ls='--',color='black',label='Applied gradient norm cap = 10')
    axes[2].set(xlabel='Global agent decision steps',ylabel='Mean pre-clipping gradient norm');axes[2].legend(fontsize=9)
    save(fig,out,'optimization_diagnostics','One native decision advances up to 4 ALE frames; gradients are logged before clipping.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--preferences-only',action='store_true',help='Render existing measured Q CSV without network inference')
    a=p.parse_args()
    if a.preferences_only:
        run=Path(a.run);data=pd.read_csv(run/'metrics/action_preferences.csv')
        q=data.pivot(index='reference',columns='action',values='q_value').to_numpy()
        names=data.sort_values('action').drop_duplicates('action').action_name.tolist()
        draw_preferences(np.load(run/'metrics/reference_rgb.npy'),q,names,run/'figures')
    else:make_plots(a.run,a.device)


if __name__=='__main__':main()
