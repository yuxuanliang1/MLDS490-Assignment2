import torch
from torch import nn


class QNetwork(nn.Module):
    """88x80 single-frame Atari CNN, with linear action-value outputs."""
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,32,8,4),nn.ReLU(),nn.Conv2d(32,64,4,2),nn.ReLU(),
                                nn.Conv2d(64,64,3,1),nn.ReLU())
        with torch.no_grad():self.flattened=int(self.conv(torch.zeros(1,1,88,80)).numel())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(self.flattened,512),nn.ReLU(),nn.Linear(512,9))

    def forward(self,images):
        if images.ndim!=4 or images.shape[1:]!=(1,88,80):
            raise ValueError(f'Expected NCHW (N,1,88,80), got {tuple(images.shape)}')
        # Compact signed input is scaled to [-1,127/128], preserving exact pixels.
        return self.head(self.conv(images.float()/128.))
