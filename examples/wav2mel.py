import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from mtts.utils.stft import TacotronSTFT
from scipy.io.wavfile import read
from mtts.utils.logging import get_logger
import librosa
import yaml
logger = get_logger(__file__)

def read_wav_np(path):
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0
    wav = wav.astype(np.float32)
    return sr, wav

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-w', '--wav_path', type=str, required=True,
                        help="root directory of wav files")
    parser.add_argument('-m', '--mel_path', type=str, required=True,
                        help="root directory of mel files")
    parser.add_argument('-d', '--device', type=str, required=False,
                        help="device, cpu or cuda:0, cuda:1,...")     
    parser.add_argument('-r', '--resample_mode', type=str, required=False,default='kaiser_fast',
                        help="use kaiser_best for high-quality audio") 

    args = parser.parse_args()
    logger.info(f'using resample mode {args.resample_mode}')
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info('loading TacotronSTFT')
    stft = TacotronSTFT(filter_length=config['fbank']['n_fft'],
                        hop_length=config['fbank']['hop_length'],
                        win_length=config['fbank']['win_length'],
                        n_mel_channels=config['fbank']['n_mels'],
                        sampling_rate=config['fbank']['sample_rate'],
                        mel_fmin=config['fbank']['fmin'],
                        mel_fmax=config['fbank']['fmax'],
                        device=args.device)

    logger.info('done')
    wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'),recursive=False)
    logger.info(f'{len(wav_files)} found in {args.wav_path}')
    mel_path = args.mel_path
    logger.info(f'mel files will be saved to {mel_path}')

    # Create all folders
    os.makedirs(mel_path, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        wav,r = librosa.load(wavpath,sr=config['fbank']['sample_rate'],res_type=args.resample_mode)
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)  # mel [1, num_mel, T]

        mel = mel.squeeze(0)  # [num_mel, T]
        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path, id), mel.numpy(), allow_pickle=False)

