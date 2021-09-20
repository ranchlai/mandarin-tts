from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mtts.models.decoder import Decoder
from mtts.models.encoder import FS2TransformerEncoder
from mtts.models.fs2_variance import VarianceAdaptor
from mtts.models.postnet import PostNet
from mtts.utils.logging import get_logger

ENCODERS = [
    FS2TransformerEncoder,
]

logger = get_logger(__file__)


def __read_vocab(file):
    with open(file) as f:
        lines = f.read().split('\n')
    lines = [line for line in lines if len(line) > 0]
    return lines


def _get_layer(emb_config: dict):  # -> Optional[List[nn.Module],List[float]]:
    logger.info(f'building embedding with config: {emb_config}')
    if emb_config['enable']:
        if emb_config['vocab'] is None:
            vocab_size = emb_config['vocab_size']
        else:
            vocab = __read_vocab(emb_config['vocab'])
            vocab_size = len(vocab)
        layer = nn.Embedding(vocab_size, emb_config['dim'], padding_idx=0)
        return layer, emb_config['weight']
    else:
        return None, None


def _build_embedding_layers(config):
    layers = nn.ModuleList()
    weights = []
    for c in (
            config['pinyin_embedding'],
            config['hanzi_embedding'],
            config['speaker_embedding'],
    ):

        layer, weight = _get_layer(c)
        if layer is not None:
            layers.append(layer)
            weights.append(weight)

    return layers, weights


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask


class FastSpeech2(nn.Module):
    """ FastSpeech2 """
    def __init__(self, config):
        super(FastSpeech2, self).__init__()

        emb_layers, emb_weights = _build_embedding_layers(config)

        EncoderClass = eval(config['encoder']['encoder_type'])
        assert EncoderClass in ENCODERS
        encoder_conf = config['encoder']['conf']
        encoder_conf.update({'emb_layers': emb_layers})
        encoder_conf.update({'embeding_weights': emb_weights})
        self.encoder = EncoderClass(**encoder_conf)

        dur_config = config['duration_predictor']
        self.variance_adaptor = VarianceAdaptor(**dur_config)

        decoder_config = config['decoder']
        self.decoder = Decoder(**decoder_config)

        n_mels = config['fbank']['n_mels']
        mel_linear_input_dim = decoder_config['hidden_dim']
        self.mel_linear = nn.Linear(mel_linear_input_dim, n_mels)
        self.postnet = PostNet(**config['postnet'], )

    def forward(self,
                input_seqs: List[Tensor],
                seq_len: Tensor,
                mel_len: Optional[Tensor] = None,
                d_target: Optional[Tensor] = None,
                max_src_len=None,
                max_mel_len=None,
                d_control=1.0,
                p_control=1.0,
                e_control=1.0):
        src_mask = get_mask_from_lengths(seq_len, max_src_len)

        if mel_len is not None:
            mel_mask = get_mask_from_lengths(mel_len, max_mel_len)
        else:
            mel_mask = None

        encoder_output = self.encoder(input_seqs, src_mask)

        if d_target is not None:
            variance_adaptor_output, d_prediction, _, _ = self.variance_adaptor(encoder_output, src_mask, mel_mask,
                                                                                d_target, max_mel_len, d_control)
        else:
            variance_adaptor_output, d_prediction, mel_len, mel_mask = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, max_mel_len, d_control)

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)

        mel_pred = self.mel_linear(decoder_output)
        postnet_input = torch.unsqueeze(mel_pred, 1)
        postnet_output = self.postnet(postnet_input) + mel_pred

        return mel_pred, postnet_output, d_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    import yaml
    with open('../../examples/aishell3/config.yaml') as f:
        config = yaml.safe_load(f)
    model = FastSpeech2(config)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
