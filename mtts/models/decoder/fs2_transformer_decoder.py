import torch.nn as nn
from torch import Tensor

from ..layers import FFTBlock, get_sinusoid_encoding_table


class FS2TransformerDecoder(nn.Module):
    """ A decoder that accepts a list of sequences as input and
    out a sequence as output.  The input and output sequences share the same length

    The input sequence is a list of tensors, which may contain text-embedding, speaker-embeddings.


    """
    def __init__(
            self,
            input_dim: int = 256,  # must ==  decoder output dim
            n_layers: int = 4,
            n_heads: int = 2,
            hidden_dim: int = 256,
            d_inner: int = 1024,
            dropout: float = 0.5,
            max_len: int = 2048,  # for computing position table
    ):
        super(FS2TransformerDecoder, self).__init__()

        self.input_dim = input_dim
        self.input_dim = input_dim

        self.max_len = max_len

        d_k = hidden_dim // n_heads
        d_v = hidden_dim // n_heads

        n_position = max_len + 1

        # self.speaker_fc = nn.Linear(512, 256, bias=False)

        pos_table = get_sinusoid_encoding_table(n_position, input_dim).unsqueeze(0)

        self.position_enc = nn.Parameter(pos_table, requires_grad=False)
        layers = []
        for _ in range(n_layers):
            layer = FFTBlock(hidden_dim, d_inner, n_heads, d_k, d_v, dropout=dropout)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, mask):
        batch_size, seq_len, input_dim = input.shape[0:3]
        if input.shape[1] != seq_len:
            raise ValueError('The input sequences must have the same length')
        if input.shape[1] != seq_len:
            raise ValueError('The input sequences must have the same dimension')

        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)

        if input.shape[1] > self.max_len:
            raise ValueError('inputs.shape[1] > self.max_len')

        pos_embed = self.position_enc[:, :seq_len, :].expand(batch_size, -1, -1)

        output = input + pos_embed
        for layer in self.layers:
            output, dec_slf_attn = layer(output, mask=mask, attn_mask=attn_mask)

        return output
