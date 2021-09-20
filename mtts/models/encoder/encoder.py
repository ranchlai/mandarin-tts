# from mtts.models.fs2_model import ENCODERS
import torch.nn as nn

from mtts.utils.logging import get_logger

from .fs2_transformer_encoder import FS2TransformerEncoder

logger = get_logger(__file__)

ENCODERS = [FS2TransformerEncoder]


class Encoder(nn.Module):
    ''' Encoder '''
    def __init__(self, encoder_type: str = 'FS2TransformerEncoder', **kwargs):

        super(Encoder, self).__init__()
        logger.info(f'building encoder with type:{encoder_type}')
        encoder_class = eval(encoder_type)
        assert encoder_class in ENCODERS
        self.config = kwargs
        self.encoder = encoder_class(**kwargs)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
