import numpy as np


def preprocess_observation(obs):
    """Exact assignment operation, returning int8 HWC; RGB sum cannot overflow."""
    obs=np.asarray(obs)
    if obs.shape!=(210,160,3) or obs.dtype!=np.uint8:
        raise ValueError(f'Expected uint8 (210,160,3), got {obs.dtype} {obs.shape}')
    image=obs[1:176:2,::2].sum(axis=2,dtype=np.int32)
    image[image==210+164+74]=0
    return (image//3-128).astype(np.int8).reshape(88,80,1)


def preprocess_chw(obs):
    return np.ascontiguousarray(preprocess_observation(obs).transpose(2,0,1))
