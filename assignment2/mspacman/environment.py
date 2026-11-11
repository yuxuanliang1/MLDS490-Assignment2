"""An exact, documented unnamespaced alias of the installed ALE v5 specification."""
from copy import deepcopy
import hashlib
import ale_py
import gymnasium as gym


def register_assignment_env():
    gym.register_envs(ale_py)
    canonical=gym.spec('ALE/MsPacman-v5')
    if 'MsPacman-v5' not in gym.registry:
        gym.register('MsPacman-v5',entry_point=canonical.entry_point,
                     kwargs=deepcopy(canonical.kwargs),max_episode_steps=canonical.max_episode_steps,
                     nondeterministic=canonical.nondeterministic)
    assignment=gym.spec('MsPacman-v5')
    if assignment.entry_point!=canonical.entry_point or assignment.kwargs!=canonical.kwargs:
        raise RuntimeError('Existing MsPacman-v5 registration does not match installed ALE/MsPacman-v5')
    return assignment


def make_env(seed=None):
    register_assignment_env()
    env=gym.make('MsPacman-v5',render_mode=None)
    if env.action_space.n!=9 or env.observation_space.shape!=(210,160,3):
        env.close();raise RuntimeError('Expected 9 actions and 210x160 RGB observations')
    if seed is not None:env.action_space.seed(seed)
    return env


def describe(env):
    path=ale_py.roms.get_rom_path('ms_pacman')
    return dict(requested_id='MsPacman-v5',actual_spec_id=env.spec.id,
        canonical_registration='ALE/MsPacman-v5',registration_alias_only=True,
        spec_kwargs=env.spec.kwargs,max_episode_steps=env.spec.max_episode_steps,
        observation_shape=env.observation_space.shape,observation_dtype=str(env.observation_space.dtype),
        actions=env.action_space.n,action_meanings=env.unwrapped.get_action_meanings(),
        rendering=None,extra_frameskip_wrapper=False,no_op_reset=False,reward_clipping=False,
        episodic_life=False,frame_stack=1,rom_file=path.name,
        rom_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rom_provenance='Already available in installed ale-py ROM package; no download/license-acceptance step performed')
