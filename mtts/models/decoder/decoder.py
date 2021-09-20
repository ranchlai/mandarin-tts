import torch.nn as nn

from mtts.models.decoder.fs2_transformer_decoder import FS2TransformerDecoder
from mtts.utils.logging import get_logger

logger = get_logger(__file__)
DECODERS = [FS2TransformerDecoder]


class Decoder(nn.Module):
    """
    """
    def __init__(self, decoder_type: str = 'FS2TransformerDecoder', **kwargs):
        super(Decoder, self).__init__()
        logger.info(f'decoder_type {decoder_type}')
        decoder_class = eval(decoder_type)
        assert decoder_class in DECODERS

        self.decoder = decoder_class(**kwargs)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
