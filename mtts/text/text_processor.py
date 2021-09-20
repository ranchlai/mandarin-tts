import torch

from mtts.datasets.dataset import Tokenizer
from mtts.utils.logging import get_logger

logger = get_logger(__file__)


class TextProcessor():
    def __init__(self, config):
        conf = config['dataset']['train']
        self.emb_tokenizers = []
        for key in conf.keys():
            if key.startswith('emb_type'):
                emb_tok = Tokenizer(conf[key]['vocab'])
                self.emb_tokenizers += [emb_tok]
                logger.info('processed emb {}'.format(conf[key]['_name']))

    def _process(self, input: str):
        segments = input.split('|')
        name = segments[0]
        segments = segments[1:]
        if len(segments) != len(self.emb_tokenizers):
            raise ValueError('Input text and emb_tokensizers are different, {segments}')

        seg_lens = [len(s.split()) for s in segments]
        n = max(seg_lens)
        # for k in seg_lens:
        # if k != n and k != 1:
        # raise ValueError(f'Input segments should share the same length, but {k}!={n} for text {input}')

        segments = [' '.join((s.split() * n)[:n]) if len(s.split()) != n else s for s in segments]
        token_tensor = []
        for seg, tokenizer in zip(segments, self.emb_tokenizers):
            tokens = tokenizer.tokenize(seg)
            token_tensor.append(torch.unsqueeze(tokens, 0))
        token_tensor = torch.cat(token_tensor, 0)
        return name, token_tensor

    def __call__(self, input):
        return self._process(input)


if __name__ == '__main__':
    import yaml
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    text_processer = TextProcessor(config)
    text = 'sil ni3 qu4 zuo4 fan4 ba5 sil|sil 你 去 做 饭 吧 sil|0 0 0 0 0 0 0'
    tensor = text_processer(text)
    print(tensor)
