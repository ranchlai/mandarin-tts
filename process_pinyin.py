import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment,get_phone
import hparams as hp

import pickle
with open('./data/name_pys_durations.pkl','rb') as F:
    pairs = pickle.load(F)



def build_from_path(in_dir, out_dir):
    index = 1
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    lines = open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8').read().split('\n')

    for line in lines:
        parts = line.strip().split('|')
        #if len(parts) ==1:
        #set_trace()
        basename = parts[0]
        text = parts[-1]

        ret = process_utterance(in_dir, out_dir, basename)
        if ret is None:
            continue
        else:
            info, f_max, f_min, e_max, e_min, n = ret

        if basename[-2:] == '66':
            val.append(info)
            #train.append(info)
        else:
            train.append(info)

        if index % 100 == 0:
            print("Done %d" % index)
        index = index + 1

        f0_max = max(f0_max, f_max)
        f0_min = min(f0_min, f_min)
        energy_max = max(energy_max, e_max)
        energy_min = min(energy_min, e_min)
        n_frames += n

    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'interval', '{}.interval'.format(basename))
    # Get alignments
    #textgrid = tgt.io.read_textgrid(tg_path)
    #phone, duration, start, end = get_alignment(
       # textgrid.get_tier_by_name('phones'))
    #phone, duration, start, end = get_phone(tg_path)
    text,duration,start,end = pairs[basename]
    
    # '{A}{B}{$}{C}', $ represents silent phones
   # text = '{' + '}{'.join(phone) + '}'
   # text = text.replace('{$}', ' ')    # '{A}{B} {C}'
   # text = text.replace('}{', ' ')     # '{A B} {C}'
    

    if start >= end:
        return None
    # Read and trim wav files
    try:
        _, wav = read(wav_path)
    except:
        print(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate,
                   frame_period=hp.hop_length/hp.sampling_rate*1000)
   
    f0 = f0[:sum(duration)]
   
    
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(
        torch.FloatTensor(wav))
   
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[
        :, :sum(duration)]
    
   
    
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None


    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename),
            duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename),
            energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename),
            mel_spectrogram.T, allow_pickle=False)
    try:
        #set_trace()
        ret = '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
       # print(ret)
    except:
        set_trace()
    return ret


def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')


in_dir = hp.data_path
out_dir = hp.preprocessed_path

#in_dir = hp.data_path
#out_dir = hp.preprocessed_path
mel_out_dir = os.path.join(out_dir, "mel")
if not os.path.exists(mel_out_dir):
    os.makedirs(mel_out_dir, exist_ok=True)
ali_out_dir = os.path.join(out_dir, "alignment")
if not os.path.exists(ali_out_dir):
    os.makedirs(ali_out_dir, exist_ok=True)
f0_out_dir = os.path.join(out_dir, "f0")
if not os.path.exists(f0_out_dir):
    os.makedirs(f0_out_dir, exist_ok=True)
energy_out_dir = os.path.join(out_dir, "energy")
if not os.path.exists(energy_out_dir):
    os.makedirs(energy_out_dir, exist_ok=True)

train, val = build_from_path(in_dir, out_dir)
write_metadata(train, val, out_dir)
