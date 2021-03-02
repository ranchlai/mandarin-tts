import os

# Dataset
wav_path = "./BZNSYP/wav"
dataset = "baker"
#data_path = "./Blizzard-2013/train/segmented/"

# Text

with_hanzi = True
hz_emb_size = 256
hz_emb_weight=0.1

# Audio and mel
### for LJSpeech ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
### for Blizzard2013 ###
#sampling_rate = 16000
#filter_length = 800
#hop_length = 200
#win_length = 800

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.25
decoder_dropout = 0.25

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000





# Quantization for F0 and energy
### for LJSpeech ###
f0_min = 0
f0_max =  1.0#796.0271999950099
energy_min = 0.0
energy_max = 1.0#176.7041015625
### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

n_bins = 256


# Checkpoints and synthesis pat
#preprocessed_path = "/dev/shm/BZNSYP/"
preprocessed_path = "./data/"

checkpoint_path = "./temp/" 
synth_path = "./synth/"
eval_path = "./eval/"
log_path = "./log/"
test_path = "./results/"


# Optimizer
batch_size = 8
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.0


# Vocoder
vocoder = 'waveglow'


# Log-scaled duration
#log_offset = 1.

#precomputed duration_mean
duration_mean = 18.877746355061273
mel_mean = -6.0304103

# Save, log and synthesis
save_step = 10000
synth_step = 3000
eval_step = 5000
eval_size = 256
log_step = 300
clear_Time = 20
