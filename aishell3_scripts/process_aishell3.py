import numpy as np
import os
from scipy.io.wavfile import read
import torch
import audio as Audio
from utils import get_alignment,get_phone
import hparams as hp
from ipdb import set_trace
import pickle
hp.dataset ='aishell3'
hz_check = open('./aishell3/vocab_hanzi.txt').read().split()
hz_check = dict([(w,True) for w in hz_check])
py_check = open('./aishell3/vocab_pinyin.txt').read().split()
py_check = dict([(w,True) for w in py_check])
def augment_cn_with_sil(py_sent,cn_sent):
    sil_loc = [i  for i,p in enumerate(py_sent.split()) if p=='sil']
    han = [h  for i,h in enumerate(cn_sent.split()) if h!='sil']

    k = 0
    final = []
    for i in range(len(han)+len(sil_loc)):
        if i in sil_loc:
            final += ['sil']
        else:
            final += [han[k]]
            k += 1
    return ' '.join(final)
def build_from_path(in_dir, out_dir):
    index = 1
    train_hanzi,val_hanzi,train_pinyin,val_pinyin = [],[],[],[]
    n_frames = 0
    lines = open('./aishell3/name_pys_hz_dur.txt').read().split('\n')
    np.random.shuffle(lines)
    for i,line in enumerate(lines):
        basename,py,hanzi,dur = line.strip().split('|')
        hanzi = augment_cn_with_sil(py,hanzi)
        #set_trace()
        py = ' '.join([w for w in py.split() if py_check.get(w,False)])
        hanzi = ' '.join([w for w in hanzi.split() if hz_check.get(w,False)])
        if len(py.split()) != len(hanzi.split()):
            set_trace()
            print(py)
            print(hanzi)
            
        if len(py.split()) != len(dur.split()):
            print(py)
            set_trace()
            print(hanzi)
        ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
        mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
        if os.path.exists(ali_filename) or os.path.exists(mel_filename):
            print('skiping '+mel_filename)
            continue
       
        name_hanzi,name_pinyin,n = process_utterance(in_dir, out_dir, basename,py,hanzi,dur)#,start,end)
        
        if basename[-3:] == '666':
            val_hanzi.append(name_hanzi)
            val_pinyin.append(name_pinyin)
            #use all for training
            train_hanzi.append(name_hanzi)
            train_pinyin.append(name_pinyin)
            
        else:
            train_hanzi.append(name_hanzi)
            train_pinyin.append(name_pinyin)

        if i % 100 == 0:
            print('processed {}/{}'.format(i,len(lines)))


    return train_hanzi,train_pinyin,val_hanzi,val_pinyin

import librosa
def process_utterance(in_dir, out_dir, basename,pinyin,hanzi,dur):#,start,end):
    wav_path = os.path.join(in_dir, '{}.wav'.format(basename))

   # start = float(start)
   # end = float(end)
    
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
 
    duration = [int(d) for d in dur.split()]
    
    
       

   # if start >= end:
        #return None
    try:
        #_, wav2 = read(wav_path)
       # set_trace()
        wav,r = librosa.load(wav_path,sr=22050)
        wav = wav*(2**15)
        wav = wav.astype('int16')
    except:
        print(wav_path)
  #  wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(
        torch.FloatTensor(wav))
   
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[
        :, :sum(duration)]
    
   
    
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None


   
    
    np.save(os.path.join(out_dir, 'alignment', ali_filename),
            duration, allow_pickle=False)

    np.save(os.path.join(out_dir, 'mel', mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    name_hanzi = '|'.join([basename, hanzi])
    name_pinyin = '|'.join([basename, pinyin])


    return name_hanzi,name_pinyin,mel_spectrogram.shape[1]




in_dir = './aishell3/wavs/'
out_dir = './aishell3/preprocessed'

#in_dir = hp.data_path
#out_dir = hp.preprocessed_path
mel_out_dir = os.path.join(out_dir, "mel")
if not os.path.exists(mel_out_dir):
    os.makedirs(mel_out_dir, exist_ok=True)
ali_out_dir = os.path.join(out_dir, "alignment")
if not os.path.exists(ali_out_dir):
    os.makedirs(ali_out_dir, exist_ok=True)


train_hanzi,train_pinyin,val_hanzi,val_pinyin = build_from_path(in_dir, out_dir)

with open(os.path.join(out_dir, 'train_hanzi.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_hanzi))

with open(os.path.join(out_dir, 'val_hanzi.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_hanzi))
with open(os.path.join(out_dir, 'train_pinyin.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_pinyin))

with open(os.path.join(out_dir, 'val_pinyin.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_pinyin))

            
            
