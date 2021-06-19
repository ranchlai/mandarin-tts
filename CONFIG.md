
## Configurations
Two config files are provided in the examples for illustration purpose. You can changed the config file if you know what you are doing. 
For example, you can remove speaker_emb from the following section, or add  prosody embedding if you have prosody label (as in biaobei dataset). 
``` yaml
dataset:
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

## Modification of config.yaml
