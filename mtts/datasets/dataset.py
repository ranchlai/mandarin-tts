import os
from typing import List, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from mtts.utils.logging import get_logger

logger = get_logger(__file__)


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]), mode='constant', constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


class Tokenizer:
    def __init__(self, vocab_file):
        if vocab_file is None:
            self.vocab = None
        else:
            self.vocab = open(vocab_file).read().split('\n')
            self.v2i = {c: i for i, c in enumerate(self.vocab)}

    def tokenize(self, text: Union[str, List]) -> Tensor:
        if self.vocab is None:  # direct mapping
            if isinstance(text, str):
                tokens = [int(t) for t in text.split()]
            else:
                tokens = [int(t) for t in text]

        else:
            if isinstance(text, str):
                tokens = [self.v2i[t] for t in text.split()]
            else:
                tokens = [self.v2i[t] for t in text]
        return torch.tensor(tokens)


def read_scp(scp_file):
    with open(scp_file, 'rt') as f:
        lines = f.read().split('\n')
    name2value = {line.split()[0]: line.split()[1:] for line in lines if len(line) > 0}
    return name2value


def check_duplicate(keys):
    key_set0 = set(keys)
    duplicate = None
    if len(keys) != len(key_set0):
        count = {k: 0 for k in key_set0}
        for k in keys:
            count[k] += 1
            if count[k] >= 2:
                duplicate = k
                break
    return duplicate
    # raise ValueError('duplicated key detected: {duplicate}')


def check_keys(*args) -> None:
    assert len(args) > 0
    for kv in args:
        dup = check_duplicate(list(kv.keys()))
        if dup:
            raise ValueError('duplicated key detected: {dup}:{kv[dup]}')

    return None


class Dataset(Dataset):
    def __init__(self, config, split='train'):
        conf = config['dataset'][split]
        self.name2wav = read_scp(conf['wav_scp'])
        self.name2mel = read_scp(conf['mel_scp'])
        self.name2dur = read_scp(conf['dur_scp'])

        self.config = config

        kv_to_check = [self.name2wav, self.name2mel, self.name2dur]

        self.emb_scps = []
        self.emb_tokenizers = []

        for key in conf.keys():
            if key.startswith('emb_type'):
                name2emb = read_scp(conf[key]['scp'])
                self.emb_scps += [name2emb]
                emb_tok = Tokenizer(conf[key]['vocab'])
                self.emb_tokenizers += [emb_tok]
                logger.info('processed emb {}'.format(conf[key]['_name']))

        kv_to_check += [name2emb]
        check_keys(*kv_to_check)

        self.names = [name for name in self.name2mel]
        mel_size = {name: os.path.getsize(self.name2mel[name][0]) for name in self.names}

        self.names = sorted(self.names, key=lambda x: mel_size[x])
        logger.info(f'Shape of longest mel: {np.load(self.name2mel[self.names[-1]][0]).shape}')
        logger.info(f'Shape of shortest mel: {np.load(self.name2mel[self.names[0]][0]).shape}')

    def __len__(self):
        return len(self.name2wav)

    def __getitem__(self, idx):
        key = self.names[idx]
        token_tensor = []
        for scp, tokenizer in zip(self.emb_scps, self.emb_tokenizers):
            emb_text = scp[key]
            tokens = tokenizer.tokenize(emb_text)
            token_tensor.append(torch.unsqueeze(tokens, 0))
        token_tensor = torch.cat(token_tensor, 0)
        mel = np.load(self.name2mel[key][0])
        if mel.shape[0] == self.config['fbank']['n_mels']:
            mel = torch.tensor(mel.T)
        else:
            mel = torch.tensor(mel)

        duration = torch.tensor([int(d) for d in self.name2dur[key]])
        return token_tensor, duration, mel


def pad_1d_tensor(x, n):
    if x.shape[0] >= n:
        return x
    x = torch.cat([x, torch.zeros((n - x.shape[0], ), dtype=x.dtype)], 0)
    return x


def pad_2d_tensor(x, n):
    if x.shape[1] >= n:
        return x
    x = torch.cat([x, torch.zeros((x.shape[0], n - x.shape[1]), dtype=x.dtype)], 1)
    return x


def pad_mel(x, n):
    if x.shape[0] >= n:
        return x
    x = torch.cat([x, torch.zeros((n - x.shape[0], x.shape[1]), dtype=x.dtype)], 0)
    return x


def collate_fn(batch):

    seq_len = []
    mel_len = []
    for (token_tensor, duration, mel) in batch:
        seq_len.append(duration.shape[-1])
        mel_len.append(mel.shape[0])

    max_seq_len = max(seq_len)
    max_mel_len = max(mel_len)
    durations = []
    token_tensors = []
    mels = []
    for token_tensor, duration, mel in batch:
        duration = pad_1d_tensor(duration, max_seq_len)
        durations.append(duration.unsqueeze_(0))
        token_tensor = pad_2d_tensor(token_tensor, max_seq_len)
        token_tensors.append(token_tensor.unsqueeze_(1))
        mel = pad_mel(mel, max_mel_len)
        mels.append(mel.unsqueeze_(0))

    durations = torch.cat(durations, 0)
    token_tensors = torch.cat(token_tensors, 1)
    mels = torch.cat(mels, 0)

    return token_tensors, durations, mels, torch.tensor(seq_len), torch.tensor(mel_len)


if __name__ == "__main__":
    import yaml
    with open('../../examples/aishell3/config.yaml') as f:
        config = yaml.safe_load(f)
    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=6, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(type(batch[-1]))
