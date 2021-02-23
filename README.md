# Chinase mandarin text to speech base on FastSpeech 2

This is an on-going work. I am still working on the details.  Code and model checkpoint will be updated in a few days. Please wait. 


由于我的业余时间比较少，只能在晚上小孩睡觉后更新。 请等待更新。

This is a modification and adpation of fastspeech 2 to mandrin. The code was  originally  implemented by https://github.com/ming024/FastSpeech2. Please refer to the origin code if you want to use it for english. 

Many modificaitons to the origin implmentation, including: 

1. Added hanzi(汉字，chinese character) embedding. It's harder for human being to read pinyin, but easier to read chinese character. Also this makes it more end-to-end. 
2. Removed pitch and energy embedding, and also the corresponding prediction network. This makes its much easier to train, especially for my gtx1060 card. I will try using them back if I have time (and hardware resources)
3. Added some tone-correcting scripts.
4. Use only waveglow in synth, as it's much better tahn melgan and griffin-lim.




# Synthesis (inference)


First clone the project and install the dependencies

```
pip3 install -r requirements.txt
```

Download [model with hanzi](https://pan.baidu.com/s/1_rB1w2YTD4BIF1OOWY9oCQ),or [model with pinyin](https://pan.baidu.com/s/1mFd1djr__qCRyCR-NKfbjQ)from baidu netdisk  and save it to ./ckpt/
- run the pinyin+hanzi model:
```
python synthesize.py --model_file ./ckpt/checkpoint_380000.pth.tar --text_file ./input.txt --channel 2 --duration_control 1.1 --output_dir ./output
```

- Or you can run pinyin model:

```
python synthesize.py --model_file ./ckpt/checkpoint_290000.pth.tar --with_hanzi 0 --text_file ./input.txt --channel 2 --duration_control 1.1 --output_dir ./output_no_hanzi
```
### Audio samples

<audio id="audio" controls="" preload="non">
<source id="wav" src="./output/waveglow_天青色等烟雨,而.wav">
</audio>





# Training

## Datasets
Currently we use baker dataset, which can be downloaded from https://www.data-baker.com/open_source.html. 

First download it from baker. 

You have to comply with the data liscense before using the data. 


## Preprocessing

I have processed the data for this experiment. You can also try 
```
python3 preprocess_pinyin.py 
python3 preprocess_hanzi.py 

```
to generate required aligments, mels, vocab for pinyin and hanzi for training. Everythin should be ready under the directory './data/'(you can change the directory in hparams.py) before training. 

## training

```
python3 train.py

```
you can monitor the log in '/home/\<user\>/.perf_logger/'

## TODO
-
# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.






