import hparams as hp
#import text
import os
from scipy.io import wavfile
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_phone(file):
    
    sil_phones = ['sil', 'sp', 'spn']
    lines=open(file).read().split('\n')[:-1]
    lines = lines[12:]
   # print(len(lines))
    start = lines[0::3]
    end = lines[1::3]
    phones = lines[2::3]
    phones = [p[1:-1] for p in phones if p[1:-1] not in sil_phones]
    start = [float(l) for l,p in zip(start,phones) if p[1:-1] not in sil_phones]
    end = [float(l) for l,p in zip(end,phones) if p[1:-1] not in sil_phones]
    durations = []
    for s,e in zip(start,end):
        durations.append(int(np.round(e*hp.sampling_rate/hp.hop_length)-np.round(s*hp.sampling_rate/hp.hop_length)))
    
    return phones,durations,start[0],end[-1]

def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(np.round(
            e*hp.sampling_rate/hp.hop_length)-np.round(s*hp.sampling_rate/hp.hop_length)))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text


    
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param



def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1)
    plt.subplots_adjust(hspace=0.4)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        spectrogram = data[i]
        axes[i].imshow(spectrogram, origin='lower')
        axes[i].set_aspect(1.3, adjustable='box')
        axes[i].set_ylim(0, hp.n_mel_channels)
        axes[i].set_title(titles[i], fontsize='medium')
     
    plt.savefig(filename, dpi=200)
    plt.close()




def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask


def get_waveglow():
    waveglow = torch.hub.load(
        'nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.to(device)

    return waveglow


def waveglow_infer(mel, waveglow, path):
    with torch.no_grad():
        wav = waveglow.infer(mel.to(device), sigma=1.0) 
        wav = wav.squeeze().cpu().numpy()
        wav = wav/np.max(np.abs(wav))*hp.max_wav_value*0.99
    wav = wav.astype('int16')
    wavfile.write(path, hp.sampling_rate, wav)


def melgan_infer(mel, melgan, path):
    with torch.no_grad():
        wav = melgan.inference(mel).cpu().numpy()
    wav = wav.astype('int16')
    wavfile.write(path, hp.sampling_rate, wav)


def get_melgan():
    melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
    melgan.eval()
    melgan.to(device)

    return melgan


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
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
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
