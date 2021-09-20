from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


# def clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Conv(nn.Module):
    """
    Convolution Module
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 w_init: str = 'linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """
    def __init__(self,
                 duration_mean: float,
                 input_dim: int = 256,
                 filter_size: int = 256,
                 kernel_size: int = 3,
                 dropout: float = 0.5):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(input_dim, filter_size, kernel_size, dropout)
        self.length_regulator = LengthRegulator()
        self.duration_mean = duration_mean

    def forward(self,
                x: Tensor,
                src_mask: Tensor,
                mel_mask: Optional[Tensor] = None,
                duration_target: Optional[Tensor] = None,
                max_len: Optional[int] = None,
                d_control: float = 1.0):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            duration_rounded = torch.clamp(torch.round((duration_target + self.duration_mean) * d_control), min=0)

            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
        else:
            # duration_rounded = torch.clamp(
            #   (torch.round(torch.exp(log_duration_prediction)-hp.log_offset)*d_control), min=0)
            duration_rounded = torch.clamp(torch.round(
                (log_duration_prediction.detach() + self.duration_mean) * d_control),
                                           min=0)
            # print('duration',duration_rounded)

            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        return x, log_duration_prediction, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """
    def __init__(self, encoder_dim: int = 256, filter_size: int = 256, kernel_size: int = 3, dropout: float = 0.5):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = filter_size
        self.kernel = kernel_size
        self.conv_output_size = filter_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            OrderedDict([("conv1d_1",
                          Conv(self.input_size,
                               self.filter_size,
                               kernel_size=self.kernel,
                               padding=(self.kernel - 1) // 2)), ("relu_1", nn.LeakyReLU()),
                         ("layer_norm_1", nn.LayerNorm(self.filter_size)), ("dropout_1", nn.Dropout(self.dropout)),
                         ("conv1d_2", Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)),
                         ("relu_2", nn.LeakyReLU()), ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                         ("dropout_2", nn.Dropout(self.dropout))]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.)

        return out
