from dataclasses import dataclass,asdict
import math


@dataclass
class Config:
    env_id:str='MsPacman-v5'
    gamma:float=.99
    seed:int=41
    episodes:int=4000
    max_steps:int=4000000
    capacity:int=50000
    warmup:int=10000
    batch_size:int=32
    learning_rate:float=1e-4
    target_interval:int=5000
    update_every:int=4
    epsilon_start:float=1.
    epsilon_end:float=.10
    epsilon_steps:int=500000
    grad_clip:float=10.
    eval_every:int=100
    eval_episodes:int=10
    checkpoint_every:int=100
    recovery_every:int=250
    replay_mode:str='random'
    reference_count:int=32
    max_hours:float=8.

    def __post_init__(self):
        if self.env_id!='MsPacman-v5':raise ValueError('Only the assignment MsPacman-v5 is supported')
        if self.gamma!=.99:raise ValueError('MsPacman main and ablations require gamma=.99')
        for key in ['episodes','max_steps','capacity','warmup','batch_size','target_interval','update_every',
                    'epsilon_steps','eval_every','eval_episodes','checkpoint_every','recovery_every','reference_count']:
            if getattr(self,key)<=0:raise ValueError(f'{key} must be positive')
        if not self.batch_size<=self.warmup<=self.capacity:raise ValueError('Require batch_size <= warmup <= capacity')
        if not 0<=self.epsilon_end<=self.epsilon_start<=1:raise ValueError('Invalid epsilon schedule')
        if self.replay_mode not in ('random','recent'):raise ValueError('Invalid replay mode')
        if any(not math.isfinite(x) or x<=0 for x in [self.learning_rate,self.grad_clip,self.max_hours]):
            raise ValueError('Learning rate, clip, and time budget must be finite and positive')

    def epsilon(self,step):
        return self.epsilon_start+min(max(step,0)/self.epsilon_steps,1)*(self.epsilon_end-self.epsilon_start)


def add_args(parser):
    for key,value in asdict(Config()).items():
        parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)


def from_args(args):return Config(**{key:getattr(args,key) for key in asdict(Config())})
