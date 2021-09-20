import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvNorm


class PostNet1d(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=7, postnet_n_convolutions=7):

        super(PostNet1d, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'), nn.InstanceNorm1d(postnet_embedding_dim)))

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size,
                             stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'), nn.InstanceNorm1d(postnet_embedding_dim)))

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim,
                         n_mel_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'), nn.InstanceNorm1d(n_mel_channels)))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.1, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.1, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
