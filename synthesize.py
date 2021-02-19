import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from string import punctuation
from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio
import numpy as np
from scipy.io import wavfile
import librosa
from hz_utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hp.with_hanzi = False
def preprocess(phone):

    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

  
def get_FastSpeech2(model_path,with_hanzi=True):
    #checkpoint_path = os.path.join(
       # hp.checkpoint_path,'no_ch_goo', "checkpoint_{}.pth.tar".format(num))
    #checkpoint_path = '/home/ranch/code/FastSpeech2/ckpt/baker/checkpoint_380000.pth.tar'
    print('loading model from',model_path)
   
    if with_hanzi:
        with open('vocab_hanzi.txt') as F:
            cn_vocab = F.read().split('\n')
    else:
        cn_vocab = None

    
    model = FastSpeech2(cn_vocab = cn_vocab)
    sd = torch.load(model_path,map_location='cpu')
    if 'model' in sd.keys(): #checkpoint file
        sd = sd['model'] # using only the model part(rather than the optim part)
    model.load_state_dict(sd)

   # model.load_state_dict(torch.load(best_model))
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, waveglow, py_text_seq,  cn_text_seq, duration_control=1.0,prefix=''):
    #sentence = sentence[:200]  # long filename will result in OS Error

    src_len = torch.from_numpy(np.array([py_text_seq.shape[1]])).to(device)
    
    mel, mel_postnet, log_duration_output, _, _, mel_len = model(
        py_text_seq, src_len, cn_seq=cn_text_seq,d_control=duration_control)
   # print(log_duration_output)
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    dst_name = os.path.join(
        '/dev/shm/', '{}-out.wav'.format(prefix))
    utils.waveglow_infer(mel_postnet_torch, waveglow, dst_name)
    return dst_name


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--text_file', type=str,default='input.txt')
    parser.add_argument('--with_hanzi', type=int, default=1)
    
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--channel', type=int, default=1)
    
    
    parser.add_argument('--output_dir', type=str, default='./output/')
    

    
    #parser.add_argument('--pitch_control', type=float, default=1.0)
    #parser.add_argument('--energy_control', type=float, default=1.0)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
        
    try:
        lines = open(args.text_file).read().split('\n')
    except:
        print('failed to open text file',args.text_file)
        exit(1)
        


    sr = 22050
    mute_len = int(22050*0.15)
    print(args)
    model = get_FastSpeech2(args.model_file,args.with_hanzi).to(device)
    hp.vocoder = 'waveglow' #force to use waveglow
    #if hp.vocoder == 'melgan':
    #melgan = utils.get_melgan()
        #melgan = Mel.load_melgan('./melgan/chkpt/aa/aa_aca5990_0100.pt','./melgan/config/default.yaml')        
    #elif hp.vocoder == 'waveglow':
    print('loading waveglow...')
    waveglow = utils.get_waveglow()
    
#     #cn_vocab1 = ['pad'] + cn_vocab + ['sil','sp1']
#     if hp.with_hanzi:
#         cn2idx = dict([(c,i) for i,c in enumerate(cn_vocab)])
#         idx2cn = dict([(i,c) for i,c in enumerate(cn_vocab)])

    torch.set_grad_enabled(False)
    
    
    for chapter in lines:
        if len(chapter)==0:
            continue
        if chapter[0] == '#':
            print('skipping ',chapter)
            continue
            
        sentences,_ = split2sent(chapter)
        audio_names = []
        for k,cn_sentence in enumerate(sentences):
            if len(cn_sentence)==0:
                continue

            print('processing',cn_sentence)
            py_sentence = convert(cn_sentence)
            py_sentence_seq = preprocess(py_sentence)
            if args.with_hanzi:
                cn_sentence_seq = convert_cn(cn_sentence)
                cn_sentence_seq = torch.from_numpy(cn_sentence_seq).long().to(device)
            else:
                cn_sentence_seq = None
            dst_name = synthesize(model, waveglow, py_sentence_seq, cn_sentence_seq,args.duration_control,prefix=cn_sentence)
            audio_names += [dst_name]
        
        #mute = np.zeros(mute_len)
        s = [librosa.load(name,sr=22050)[0] for name in audio_names]
        s = [_s[mute_len:-mute_len]/np.max(np.abs(_s)) for _s in s]
        s = np.concatenate(s)
        s = s/np.max(np.abs(s))*0.99
        fn = '/dev/shm/{}_{}_22k.wav'.format(hp.vocoder,chapter[:8])

       # print(fn)
        wavfile.write(fn,22050,(s*32767).astype('int16'))
        final_fn = os.path.join(args.output_dir,'{}_{}.wav'.format(hp.vocoder,chapter[:8]))
        cmd = 'ffmpeg -i {} -ac {} -ar 48000 -strict -2 {} -y'.format(fn,args.channel,final_fn)
        os.system(cmd)
        print('audio written to {}'.format(final_fn))


    
    
#     utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], [
#                     'Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))
