import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .evaluate import load_agent


plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 220, 'axes.grid': True, 'grid.alpha': .2})


def save(fig, output, name, note):
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(output / f'{name}.png')
    fig.savefig(output / f'{name}.pdf')
    plt.close(fig)


def make_plots(run, checkpoint, evaluation, device='cpu'):
    run, evaluation = Path(run), Path(evaluation)
    output = run / 'figures'; output.mkdir(exist_ok=True)
    episodes = pd.read_csv(run / 'logs/episodes.csv')
    periodic = pd.read_csv(run / 'logs/periodic_evaluation.csv')
    results = pd.read_csv(evaluation / 'episodes.csv')
    plot_config = load_agent(checkpoint, device)[0].config
    gamma = plot_config.gamma
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(episodes.episode, episodes.initial_state_max_q, color='#236e9a', lw=1.5)
    ax.axhline(1/(1-gamma), ls='--', color='#ae5a2a', label=f'Continuing +1 reward scale: 1/(1−γ) = {1/(1-gamma):.1f}')
    ax.set(xlabel='Training episode', ylabel='maxₐ Q_online(initial observation, a)',
           title='Initial-state maximum predicted Q-value (end-of-episode network)')
    ax.legend(fontsize=9)
    save(fig,output,'initial_state_max_q', 'Each point uses that episode’s initial observation and final online weights. Discounted Q is not a 500-step score.')
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(episodes.episode, episodes.reward, alpha=.35, lw=.8, color='#236e9a', label='Exploratory training reward')
    ax.plot(episodes.episode, episodes.reward.rolling(100,min_periods=100).mean(), color='#173a55', lw=2, label='Trailing 100-episode mean')
    ax.plot(periodic.episode, periodic['mean'], 'o-', ms=3, color='#b45121', label=f'Greedy selection-panel mean ({plot_config.eval_episodes} episodes)')
    ax.axhline(500, ls='--', color='black', label='Native maximum: 500')
    ax.set(xlabel='Training episode',ylabel='Undiscounted reward (steps)',title='CartPole-v1 training and periodic greedy evaluation',ylim=(0,525))
    ax.legend(fontsize=9)
    save(fig,output,'training_rewards','Moving average first appears at episode 100. Greedy selection scores use separate, fixed seeds.')
    fig, ax = plt.subplots(figsize=(9,5))
    r = results.reward
    ax.hist(r, bins=np.linspace(0,500,26), color='#236e9a', edgecolor='white')
    ax.axvline(r.mean(), color='#b45121', lw=2, label=f'Mean = {r.mean():.3f}; SD (population) = {r.std(ddof=0):.3f}')
    ax.set(xlabel='Greedy episode undiscounted reward (steps)',ylabel='Number of episodes',title='Exactly 500 held-out greedy evaluation episodes')
    ax.legend(fontsize=10)
    early = r[r < 500]
    if len(early) and (r == 500).mean() >= .9:
        # Keep rare failures visible beside the dominant full-score bin.
        inset = ax.inset_axes([.08, .48, .43, .32])
        inset.hist(early, bins=np.linspace(0,500,26), color='#b45121', edgecolor='white')
        inset.set(title=f'Early failures only (n={len(early)})', xlim=(0,500),
                  ylabel='Count', xlabel='Reward (steps)')
        inset.title.set_fontsize(9)
        inset.xaxis.label.set_size(8); inset.yaxis.label.set_size(8)
        inset.tick_params(labelsize=8)
        inset.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    save(fig,output,'evaluation_histogram',f'Perfect scores: {(r==500).sum()}/500 ({(r==500).mean():.1%}). Distinct reset seeds; ε = 0.')
    agent, _ = load_agent(checkpoint, device)
    angles = np.linspace(-.2094,.2094,161)
    velocities = np.linspace(-2,2,161)
    theta, omega = np.meshgrid(angles, velocities)
    states = np.zeros((theta.size,4),np.float32)
    states[:,2], states[:,3] = theta.ravel(), omega.ravel()
    q = agent.q_values(states)
    delta = (q[:,1]-q[:,0]).reshape(theta.shape)
    actions = delta > 0
    pd.DataFrame(dict(theta_rad=states[:,2],omega_rad_s=states[:,3],q_left=q[:,0],q_right=q[:,1],action=q.argmax(1))).to_csv(run/'metrics/policy_grid.csv',index=False)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.pcolormesh(theta, omega, actions, cmap=ListedColormap(['#aacce2','#efb58d']), shading='auto',vmin=0,vmax=1)
    contours = ax.contour(theta,omega,delta,levels=[-1,-.5,0,.5,1],colors='#293b49',linewidths=.8)
    ax.clabel(contours,inline=True,fontsize=8,fmt='%g')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#aacce2',label='Push left (0)'),Patch(color='#efb58d',label='Push right (1)')],fontsize=10)
    ax.set(xlabel='Pole angle θ (rad)',ylabel='Pole angular velocity ω (rad/s)',title='Greedy action slice and Q(right) − Q(left) contours')
    save(fig,output,'policy_action_map','Cart position = 0 m; cart velocity = 0 m/s. A fixed slice of a four-dimensional policy, not its complete behavior.')
    fig, axes = plt.subplots(2,1,figsize=(9,7),sharex=True)
    axes[0].plot(episodes.episode,episodes.mean_loss,color='#236e9a'); axes[0].set(ylabel='Mean Huber loss',title='Optimization diagnostics')
    axes[1].plot(episodes.episode,episodes.epsilon,color='#b45121'); axes[1].set(xlabel='Training episode',ylabel='Behavior ε')
    save(fig,output,'optimization_diagnostics','Epsilon follows global environment steps. Episode losses average optimizer updates within each episode.')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--evaluation',required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args(); make_plots(a.run,a.checkpoint,a.evaluation,a.device)


if __name__ == '__main__': main()
