# Chinese mandarin text to speech (MTTS)

This is a modularized Text-to-speech framework aiming to support fast research and product developments. Main features include
- all modules are configurable via yaml, 
- speaker embedding / prosody embeding/ multi-stream  text embedding are supported and configurable, 
- various vocoders (VocGAN, hifi-GAN, waveglow, melGAN) are supported by adapter so that comparison across different vocoders can be done easily, 
- durations/pitch/energy variance predictor are supported, and other variances can be added easily, 
- and more on the road-map. 

Contributions are welcome. 

### Audio samples

- Interesting audio samples for aishell3 added [here](./docs/samples/aishell3).
- The <a href="https://ranchlai.github.io/mandarin-tts/">github page</a> also hosts some samples for both [biaobei](https://www.data-baker.com/en/#/data/index/source) and [aishell3](https://www.openslr.org/93/).

## Quick start

### Install

```
git clone https://github.com/ranchlai/mandarin-tts.git
cd mandarin-tts
git submodule update --force --recursive --init --remote
pip install -e .  # this is necessary!

```

### Training
Two examples are provided here: [./examples/biaobei)](./examples/biaobei) and [./examples/aishell3](./examples/aishell3).

To train your own models, first make a copy from existing examples, then  prepare the melspectrogram features using [./examples/wav2mel.py](./examples/wav2mel.py) by
``` sh
cd examples
python wav2mel.py -c ./aishell3/config.yaml -w <aishell3_wav_folder> -m <mel_folder> -d cpu
```

prepare the scp files necessary for training, 
``` sh
cd examples/aishell3
python prepare.py --wav_folder <aishell3_wav_folder>  --mel_folder <mel_folder> --dst_folder ./train/
```
This will generate scp files required by config.yaml (in the dataset/train section). 
You would also need to check that everything is fine in the [config file](./examples/aishell3/config.yaml). 
Usually you don't need to change the code. 

Now you can start your training by 
``` sh
cd examples/aishell3
python ../../mtts/train.py -c config.yaml -d cuda
```

For biaobei dataset, the workflow is the same, except that there is no speaker embedding but you can add prosody embedding. 

### Synthesize 
``` sh
python ../../mtts/synthesize.py  -d cuda --c config.yaml --checkpoint ./checkpoints/checkpoint_1240000.pth.tar -i input.txt
```
Checkpoints to get started: [aishell3](https://zenodo.org/record/4912321#.YMN2-FMzakA) and [Biaobei](https://zenodo.org/record/4910507#.YMN29lMzakA)

<b> About input text </b>
The [input.txt]('./examples/aishell3/input.txt) should be consistent with your setting  of emb_type1 to emb_type_n in config file, i.e., same type, same order.

To facilitate transcription of hanzi to pinyin, you can try:
```
cd examples/aishell3/
python ../../mtts/text/gp2py.py -t "为适应新的网络传播方式和读者阅读习惯"
>> sil wei4 shi4 ying4 xin1 de5 wang3 luo4 chuan2 bo1 fang1 shi4 he2 du2 zhe3 yue4 du2 xi2 guan4 sil|sil 为 适 应 新 的 网 络 传 播 方 式 和 读 者 阅 读 习 惯 sil
```
Not you can copy the text to [input.txt](./examples/aishell3/input.txt), and remember to put down the self-defined name and speaker id, separated by '|'. 

If goes well, audio examples can be found [here](./examples/aishell3/outputs/) and [here](./examples/biaobei/outputs/)

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



