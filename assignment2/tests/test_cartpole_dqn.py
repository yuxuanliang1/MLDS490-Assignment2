from dataclasses import replace
import numpy as np
import pytest
import torch
import gymnasium as gym
from assignment2.cartpole.config import Config, seed_all
from assignment2.cartpole.dqn import DQN, bellman_target
from assignment2.cartpole.replay_buffer import ReplayBuffer
from assignment2.cartpole.train import save_checkpoint
from assignment2.cartpole.evaluate import load_agent
from assignment2.cartpole.evaluate import evaluate_policy


def buffer(n=8, capacity=8):
    b = ReplayBuffer(capacity, 3)
    for i in range(n):
        b.add(np.full(4, i, np.float32), i % 2, 1., np.full(4, i+1, np.float32), i % 3 == 0, i % 5 == 0)
    return b


def test_underflow_wraparound_and_random_sampling():
    b = buffer(0, 4)
    with pytest.raises(ValueError): b.sample(1)
    b = buffer(7, 4)
    assert len(b) == 4
    assert set(b.sample(4)['state'][:, 0]) == {3, 4, 5, 6}
    assert b.sample(3, 'recent')['state'][:, 0].tolist() == [4, 5, 6]
    with pytest.raises(ValueError): b.sample(5)
    samples = [tuple(b.sample(2)['state'][:, 0]) for _ in range(30)]
    assert len(set(samples)) > 4


def test_bellman_termination_and_time_limit_bootstrap():
    agent = DQN(Config())
    with torch.no_grad():
        for p in agent.target.parameters(): p.zero_()
        agent.target.layers[-1].bias.copy_(torch.tensor([2., 4.]))
    # Both flags may be true: genuine termination always takes precedence.
    targets = bellman_target(agent.target, torch.zeros(4, 4), torch.ones(4),
                             torch.tensor([True, False, False, True]), .95)
    np.testing.assert_allclose(targets, [1., 4.8, 4.8, 1.])
    assert not targets.requires_grad


def test_target_frozen_and_sync_only_on_update_interval():
    seed_all(3)
    agent = DQN(replace(Config(), target_interval=2))
    old = {k:v.clone() for k,v in agent.target.state_dict().items()}
    agent.update(buffer().sample(8))
    assert all(torch.equal(old[k], v) for k,v in agent.target.state_dict().items())
    assert any(not torch.equal(old[k], v) for k,v in agent.online.state_dict().items())
    assert all(p.grad is None for p in agent.target.parameters())
    agent.update(buffer().sample(8))
    assert all(torch.equal(v, agent.online.state_dict()[k]) for k,v in agent.target.state_dict().items())


def test_soft_update_weights_old_target_not_online():
    agent=DQN(replace(Config(),target_mode='soft',target_tau=.999))
    old={k:v.clone() for k,v in agent.target.state_dict().items()}
    agent.update(buffer().sample(8))
    for key,value in agent.target.state_dict().items():
        expected=.999*old[key]+.001*agent.online.state_dict()[key]
        torch.testing.assert_close(value,expected,rtol=1e-6,atol=1e-8)
    assert agent.target_updates==1
    assert all(p.grad is None for p in agent.target.parameters())


def test_selected_action_indexing_and_huber_reference():
    agent=DQN(Config())
    with torch.no_grad():
        for network in (agent.online,agent.target):
            for p in network.parameters():p.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([2.,4.]))
    b=buffer().sample(8); b['terminated'][:]=True
    expected=np.where(b['action']==0,.5,2.5).mean() # Huber(errors 1 and 3)
    loss,_,_=agent.update(b)
    assert loss==pytest.approx(expected)


def test_evaluation_does_not_change_training_environment_or_weights():
    agent=DQN(Config());env=gym.make('CartPole-v1')
    try:
        env.reset(seed=44);before=np.array(env.unwrapped.state).copy()
        weights={k:v.clone() for k,v in agent.online.state_dict().items()}
        frame=evaluate_policy(agent,[45,46],batch_envs=2)
        assert len(frame)==2 and frame.seed.nunique()==2
        np.testing.assert_array_equal(env.unwrapped.state,before)
        assert all(torch.equal(weights[k],v) for k,v in agent.online.state_dict().items())
        assert agent.updates==0
    finally:env.close()


def test_epsilon_extremes_and_step_schedule():
    a = DQN(Config())
    rng = np.random.default_rng(5)
    q_action = int(a.q_values(np.zeros(4, np.float32)).argmax())
    assert {a.act(np.zeros(4), 0., rng) for _ in range(100)} == {q_action}
    actions = [a.act(np.zeros(4), 1., rng) for _ in range(1000)]
    assert .43 < np.mean(actions) < .57
    assert a.config.epsilon(0) == 1.
    assert a.config.epsilon(10000) == pytest.approx(.525)
    assert a.config.epsilon(20000) == pytest.approx(.05)
    assert a.config.epsilon(40000) == pytest.approx(.05)


def test_checkpoint_reload_consistency(tmp_path):
    a = DQN(Config()); a.update(buffer().sample(8))
    path = tmp_path / 'model.pt'
    save_checkpoint(path, a, 1, 8, 10.)
    loaded, ckpt = load_agent(path, 'cpu')
    states = np.random.default_rng(7).normal(size=(30,4)).astype(np.float32)
    np.testing.assert_array_equal(a.q_values(states), loaded.q_values(states))
    assert ckpt['optimizer_updates'] == 1


@pytest.mark.parametrize('source', ['reward', 'online', 'target'])
def test_nonfinite_detection_before_update(source):
    a = DQN(Config()); b = buffer().sample(8)
    before = {k:v.clone() for k,v in a.online.state_dict().items()}
    if source == 'reward': b['reward'][0] = np.inf
    else:
        with torch.no_grad():
            getattr(a, source).layers[-1].bias[0] = float('nan')
    with pytest.raises(FloatingPointError): a.update(b)
    assert a.updates == 0
    if source == 'reward':
        assert all(torch.equal(before[k],v) for k,v in a.online.state_dict().items())


def test_immediate_failure_includes_final_reward():
    env = gym.make('CartPole-v1')
    try:
        # Supported reset bounds put all four state components at 0.25;
        # initial pole angle exceeds the true termination threshold.
        state, _ = env.reset(seed=8, options={'low': .25, 'high': .25})
        _, reward, terminated, truncated, _ = env.step(0)
        assert terminated and not truncated and reward == 1.
    finally: env.close()


def test_real_500_step_time_limit_and_reward_count():
    env = gym.make('CartPole-v1')
    try:
        state, _ = env.reset(seed=0)
        total = 0
        for step in range(1, 501):
            # Analytic feedback controller solely tests environment bookkeeping.
            action = int(state[2] + .2 * state[3] > 0)
            state, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated: break
        assert step == 500 and total == 500 and truncated and not terminated
    finally: env.close()


@pytest.mark.parametrize('kwargs', [{'env_id':'CartPole-v0'}, {'capacity':4}, {'target_interval':0}, {'gamma':1.}, {'epsilon_end':2.}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)
