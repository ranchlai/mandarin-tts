# Introduction to configurations

## Dataset
``` yaml
dataset: # the dataset part is for training only
  train:
    wav_scp: './train/wav.scp'
    mel_scp: './train/mel.scp'
    dur_scp: './train/dur.scp'
    emb_type1:
      _name: 'pinyin'
      scp: './train/py.scp'
      vocab: 'py.vocab'
    emb_type2:
      _name: 'graphic'
      scp: './train/gp.scp'
      vocab: 'gp.vocab'
    #emb_type3:
      #_name: 'speaker'
     # scp: './train/spk.scp'
     # vocab: # dosn't need vocab
    emb_type4:
      _name: 'prosody'
      scp: './train/psd.scp'
      vocab:
```

## Vocoder
```yaml
vocoder:
  type: VocGan # choose one of the following
  MelGAN:
    checkpoint: ~/checkpoints/melgan/melgan_ljspeech.pth
    config: ~/checkpoints/melgan/default.yaml
    device: cpu
  VocGan:
    checkpoint: ~/checkpoints/vctk_pretrained_model_3180.pt #~/checkpoints/ljspeech_29de09d_4000.pt
    denoise: True
    device: cpu
  HiFiGAN:
    checkpoint: ~/checkpoints/VCTK_V3/generator_v3  # you need to download checkpoint and set the params here
    device: cpu
  Waveglow:
    checkpoint:  ~/checkpoints/waveglow_256channels_universal_v5_state_dict.pt
    sigma: 1.0
    denoiser_strength: 0.0 # try 0.1
    device: cpu #try cpu if out of memory

```


## Make your own changes
Two config files are provided in the examples for illustration purpose. You can changed the config file if you know what you are doing.
For example, you can remove speaker_emb from the following section, or add  prosody embedding if you have prosody label (as in biaobei dataset).
