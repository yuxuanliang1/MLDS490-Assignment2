from dataclasses import replace
import numpy as np
import pytest
import torch
from assignment2.mspacman.config import Config
from assignment2.mspacman.model import QNetwork
from assignment2.mspacman.agent import DQN,bellman_target,save_model,load_agent
from assignment2.mspacman.preprocessing import preprocess_observation,preprocess_chw
from assignment2.mspacman.replay_buffer import ReplayBuffer
from assignment2.mspacman.runtime import seed_all
from assignment2.mspacman.environment import make_env,describe


@pytest.fixture(autouse=True)
def one_cpu_thread():torch.set_num_threads(1)


def images(value=0):return np.full((1,88,80),value,np.int8)


def replay(n=8,capacity=8):
    buffer=ReplayBuffer(capacity,7)
    for i in range(n):buffer.add(images(i),i%9,float(i),images(i+1),i%3==0,i%4==0)
    return buffer


def test_exact_preprocessing_overflow_mask_and_layout():
    rgb=np.full((210,160,3),255,np.uint8)
    assert (preprocess_observation(rgb)==127).all() # 255+255+255 must not overflow
    rgb[:]=[210,164,74]
    assert (preprocess_observation(rgb)==-128).all()
    rgb=np.random.default_rng(2).integers(0,256,size=(210,160,3),dtype=np.uint8)
    reference=rgb[1:176:2,::2].sum(axis=2);reference[reference==448]=0
    reference=(reference//3-128).astype(np.int8).reshape(88,80,1)
    np.testing.assert_array_equal(preprocess_observation(rgb),reference)
    np.testing.assert_array_equal(preprocess_chw(rgb),reference.transpose(2,0,1))
    with pytest.raises(ValueError):preprocess_observation(rgb.astype(np.float32))


def test_replay_underflow_wraparound_flags_and_storage():
    b=replay(0,4)
    with pytest.raises(ValueError):b.sample(1)
    b=replay(7,4)
    batch=b.sample(4)
    assert set(batch['state'][:,0,0,0])=={3,4,5,6}
    assert b.sample(3,'recent')['state'][:,0,0,0].tolist()==[4,5,6]
    assert b.data['state'].dtype==np.int8 and b.data['next_state'].dtype==np.int8
    assert sum(array.nbytes for array in b.data.values())==ReplayBuffer.memory_bytes(4)
    np.testing.assert_array_equal(batch['next_state'],batch['state']+1)
    np.testing.assert_array_equal(batch['terminated'],batch['reward']%3==0)
    with pytest.raises(ValueError):b.add(images(),9,1,images(),False,False)
    with pytest.raises(FloatingPointError):b.add(images(),0,np.inf,images(),False,False)


def test_replay_restore_includes_rng_and_ring_position(tmp_path):
    b=replay(15,8);path=tmp_path/'replay.npz';b.save(path);loaded=ReplayBuffer.load(path)
    assert loaded.position==b.position and len(loaded)==len(b)
    # Compare one common minibatch, not a fresh RNG draw for each key.
    b.save(path);loaded=ReplayBuffer.load(path);left=b.sample(5);right=loaded.sample(5)
    for key in left:np.testing.assert_array_equal(left[key],right[key])


def test_network_dimensions_and_action_indexed_huber_reference():
    agent=DQN(Config());assert agent.online.flattened==2688
    assert agent.q_values(np.zeros((3,1,88,80),np.int8)).shape==(3,9)
    with torch.no_grad():
        for net in (agent.online,agent.target):
            for p in net.parameters():p.zero_()
        agent.online.head[-1].bias.copy_(torch.arange(9,dtype=torch.float32))
    batch=replay().sample(8);batch['reward'][:]=0;batch['terminated'][:]=True
    err=batch['action'].astype(float)
    expected=np.where(err<1,.5*err**2,err-.5).mean()
    assert agent.update(batch)['loss']==pytest.approx(expected)


def test_bellman_true_termination_only_and_target_frozen():
    agent=DQN(replace(Config(),target_interval=2))
    with torch.no_grad():
        for p in agent.target.parameters():p.zero_()
        agent.target.head[-1].bias.copy_(torch.arange(9,dtype=torch.float32))
    target=bellman_target(agent.target,torch.zeros(3,1,88,80,dtype=torch.int8),torch.ones(3),
                          torch.tensor([True,False,False]),.99)
    torch.testing.assert_close(target,torch.tensor([1.,8.92,8.92]))
    assert not target.requires_grad
    before={k:v.clone() for k,v in agent.target.state_dict().items()}
    agent.update(replay().sample(8))
    assert all(torch.equal(before[k],v) for k,v in agent.target.state_dict().items())
    assert all(p.grad is None for p in agent.target.parameters())
    agent.update(replay().sample(8))
    assert all(torch.equal(v,agent.online.state_dict()[k]) for k,v in agent.target.state_dict().items())


def test_epsilon_extremes_and_step_schedule():
    a=DQN(Config());rng=np.random.default_rng(10);state=images()
    expected=int(a.q_values(state).argmax())
    assert {a.act(state,0,rng) for _ in range(10)}=={expected}
    actions=[a.act(state,1,rng) for _ in range(900)]
    counts=np.bincount(actions,minlength=9)
    assert (counts>65).all() and (counts<140).all()
    assert a.config.epsilon(0)==1 and a.config.epsilon(250000)==pytest.approx(.55)
    assert a.config.epsilon(900000)==pytest.approx(.1)


def test_checkpoint_and_nonfinite_loss(tmp_path):
    a=DQN(Config());batch=replay().sample(8);a.update(batch)
    path=tmp_path/'model.pt';save_model(path,a,episode=1)
    loaded,_=load_agent(path,'cpu')
    np.testing.assert_array_equal(a.q_values(images()),loaded.q_values(images()))
    batch['reward'][0]=float('inf')
    updates=a.updates
    with pytest.raises(FloatingPointError):a.update(batch)
    assert a.updates==updates


def test_exact_environment_alias_rgb_actions_and_short_timelimit():
    import gymnasium as gym
    env=make_env(9)
    try:
        description=describe(env)
        assert env.spec.id=='MsPacman-v5' and env.action_space.n==9
        assert description['spec_kwargs']['frameskip']==4
        assert description['spec_kwargs']['repeat_action_probability']==.25
        limited=gym.wrappers.TimeLimit(env,max_episode_steps=9)
        obs,_=limited.reset(seed=9)
        assert preprocess_chw(obs).dtype==np.int8
        for action in range(9):obs,r,term,trunc,info=limited.step(action)
        assert trunc and not term
    finally:env.close()


def test_native_ale_frame_limit_and_alias_equivalence():
    import gymnasium as gym
    alias=make_env(19);canonical=gym.make('ALE/MsPacman-v5',render_mode=None)
    limited=gym.make('MsPacman-v5',max_num_frames_per_episode=12,render_mode=None)
    try:
        left,_=alias.reset(seed=19);right,_=canonical.reset(seed=19)
        np.testing.assert_array_equal(left,right)
        for action in np.random.default_rng(19).integers(9,size=120):
            l=alias.step(int(action));r=canonical.step(int(action))
            np.testing.assert_array_equal(l[0],r[0]);assert l[1:4]==r[1:4]
            if l[2] or l[3]:break
        limited.reset(seed=19)
        for i in range(3):_,reward,term,trunc,_=limited.step(0)
        assert trunc and not term
    finally:alias.close();canonical.close();limited.close()


def test_nonfinite_q_guard_and_memory_guard(monkeypatch):
    agent=DQN(Config())
    with torch.no_grad():agent.online.head[-1].bias[0]=float('nan')
    with pytest.raises(FloatingPointError):agent.q_values(images())
    import assignment2.mspacman.replay_buffer as module
    class AlmostNoMemory:available=1
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:AlmostNoMemory())
    with pytest.raises(MemoryError):ReplayBuffer(50000)


def test_full_recovery_preserves_optimizer_replay_and_rng(tmp_path):
    import random
    from assignment2.mspacman.checkpoint import save_recovery,load_recovery
    from assignment2.mspacman.runtime import write_json
    for sub in ['replay','checkpoints','metrics']:(tmp_path/sub).mkdir()
    write_json(tmp_path/'metrics/selection.json',dict(checkpoint='unused.pt',episode=2,mean=100.))
    seed_all(77);config=replace(Config(),capacity=8,warmup=8,batch_size=8,target_interval=2)
    agent=DQN(config);buffer=replay();behavior=np.random.default_rng(99)
    agent.update(buffer.sample(8));agent.update(buffer.sample(8))
    path=save_recovery(tmp_path,agent,buffer,behavior,2,20,80,1.,100.,'reference')
    expected_batch=buffer.sample(4);expected_action=behavior.integers(9)
    expected_torch=torch.rand(3);expected_numpy=np.random.random(3);expected_python=random.random()
    agent.update(expected_batch)
    restored,restored_buffer,metadata=load_recovery(path,tmp_path,'cpu')
    restored_behavior=np.random.default_rng();restored_behavior.bit_generator.state=metadata['behavior_rng']
    actual_batch=restored_buffer.sample(4)
    for key in expected_batch:np.testing.assert_array_equal(expected_batch[key],actual_batch[key])
    assert restored_behavior.integers(9)==expected_action
    torch.testing.assert_close(torch.rand(3),expected_torch,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.random(3),expected_numpy);assert random.random()==expected_python
    restored.update(actual_batch)
    for key,value in agent.online.state_dict().items():torch.testing.assert_close(value,restored.online.state_dict()[key],rtol=0,atol=0)


def test_resume_rolls_back_later_log_rows_but_preserves_archive(tmp_path):
    import csv
    import json
    from assignment2.mspacman.checkpoint import rollback_logs
    for sub in ['logs','metrics','checkpoints']:(tmp_path/sub).mkdir()
    for filename,key in [('episodes.csv','episode'),('validation.csv','episode'),('updates.csv','step'),('transitions.csv','step')]:
        with (tmp_path/'logs'/filename).open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow([key,'value']);writer.writerows([[1,10],[2,20],[3,30]])
    (tmp_path/'metrics/validation_ep00003.csv').write_text('seed,reward\n1,999\n')
    archive=rollback_logs(tmp_path,dict(episode=2,steps=2,selection=dict(checkpoint='saved.pt',episode=2,mean=20)),1)
    for path in (tmp_path/'logs').glob('*.csv'):
        assert len(list(csv.DictReader(path.open())))==2
        assert len(list(csv.DictReader((archive/path.name).open())))==3
    assert (archive/'validation_ep00003.csv').exists()
    assert json.loads((tmp_path/'metrics/selection.json').read_text())['checkpoint']=='saved.pt'
