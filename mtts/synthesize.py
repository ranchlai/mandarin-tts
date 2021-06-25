import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from mtts.models.fs2_model import FastSpeech2
from mtts.models.vocoder import *
from mtts.text import TextProcessor
from mtts.utils.logging import get_logger

logger = get_logger(__file__)


def check_ffmpeg():
    r, path = subprocess.getstatusoutput("which ffmpeg")
    return r == 0


with_ffmpeg = check_ffmpeg()


def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model


def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input.txt')
    parser.add_argument('--duration', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--checkpoint', type=str, required=True, default='')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', choices=['cuda', 'cpu'], type=str, default='cuda')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    sr = config['fbank']['sample_rate']

    vocoder = build_vocoder(args.device, config)
    text_processor = TextProcessor(config)
    model = FastSpeech2(config)

    if args.checkpoint != '':
        sd = torch.load(args.checkpoint, map_location=args.device)
        if 'model' in sd.keys():
            sd = sd['model']
    model.load_state_dict(sd)
    del sd  # to save mem
    model = model.to(args.device)
    torch.set_grad_enabled(False)

    try:
        lines = open(args.input).read().split('\n')
    except:
        print('Failed to open text file', args.input)
        print('Treating input as text')
        lines = [args.input]

    for line in lines:
        if len(line) == 0 or line.startswith('#'):
            continue
        logger.info(f'processing {line}')
        name, tokens = text_processor(line)
        tokens = tokens.to(args.device)
        seq_len = torch.tensor([tokens.shape[1]])
        tokens = tokens.unsqueeze(1)
        seq_len = seq_len.to(args.device)
        max_src_len = torch.max(seq_len)
        output = model(tokens, seq_len, max_src_len=max_src_len, d_control=args.duration)
        mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

        # convert to waveform using vocoder
        mel_postnet = mel_postnet[0].transpose(0, 1).detach()
        mel_postnet += config['fbank']['mel_mean']
        wav = vocoder(mel_postnet)
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join(args.output_dir, f'{name}.wav')
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        logger.info(f'writing file to {dst_file}')
        wavfile.write(dst_file, sr, wav)
