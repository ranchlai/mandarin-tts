from typing import List

import torch.nn as nn
from torch import Tensor

from ..layers import FFTBlock, get_sinusoid_encoding_table


class FS2TransformerEncoder(nn.Module):
    ''' FS2TransformerEncoder '''
    def __init__(
        self,
        emb_layers: nn.ModuleList,
        embeding_weights: List[float],
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 2,
        d_inner: int = 1024,
        dropout: float = 0.5,
        max_len: int = 1024,
    ):

        super(FS2TransformerEncoder, self).__init__()

        self.emb_layers = emb_layers
        self.embeding_weights = embeding_weights
        self.hidden_dim = hidden_dim

        d_k = hidden_dim // n_heads
        d_v = hidden_dim // n_heads
        d_model = hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = FFTBlock(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            self.layers.append(layer)

    def forward(self, texts: List[Tensor], mask: Tensor):

        if len(self.embeding_weights) != len(texts):
            raise ValueError(f'Input texts has length {len(texts)},　\
                    but embedding module list has length {len(self.embeding_weights)}')
        batch_size = texts[0].shape[0]
        seq_len = texts[0].shape[1]
        attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        text_embed = self.emb_layers[0](texts[0]) * self.embeding_weights[0]
        n_embs = len(self.embeding_weights)
        for i in range(1, n_embs):
            text_embed += self.emb_layers[i](texts[i]) * self.embeding_weights[i]

        if self.training:
            pos_embed = get_sinusoid_encoding_table(seq_len, self.hidden_dim)
            assert pos_embed.shape[0] == seq_len
            pos_embed = pos_embed[:seq_len, :]
            pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
            pos_embed = pos_embed.to(texts[0].device)
        else:
            pos_embed = self.position_enc[:, :seq_len, :]
            pos_embed = pos_embed.expand(batch_size, -1, -1)

        all_embed = text_embed + pos_embed

        for layer in self.layers:
            all_embed, enc_slf_attn = layer(all_embed, mask=mask, attn_mask=attn_mask)

        return all_embed
