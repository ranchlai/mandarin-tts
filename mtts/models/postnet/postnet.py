import torch.nn as nn

from .postnet_1d import PostNet1d
from .postnet_unet import PostUNet

POSTNETS = [PostNet1d, PostUNet]


class PostNet(nn.Module):
    """
    Interface class for postnets
    """
    def __init__(self, postnet_type: str = 'PostUNet', **kwargs):

        super(PostNet, self).__init__()
        PostNetClass = eval(postnet_type)
        assert PostNetClass in POSTNETS
        self.postnet = PostNetClass(**kwargs)
        self.config = kwargs

    def forward(self, x, **kwargs):
        return self.postnet(x, **kwargs)
