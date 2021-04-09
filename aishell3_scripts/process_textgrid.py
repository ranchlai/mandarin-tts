import textgrid
import tqdm
import glob
import hparams as hp
import numpy as np

def get_pys_dur(f):
    tg = textgrid.TextGrid.fromFile(f)
    dur = [int(np.round(inter.duration()*hp.sampling_rate/hp.hop_length))  for inter in tg[0]]
    py = ['sil' if inter.mark=='' else inter.mark  for inter in tg[0]]
    return py,dur
    #print(inter.duration(),inter.mark)
    
    
lines = open('./aishell3/content.txt').read().split('\n')
name2cn = dict([(l.split('.wav')[0],' '.join(['sil']+l.split('\t')[-1].split(' ')[::2]+['sil'])) for l in lines])

files = glob.glob('./aishell3/wavs/*.TextGrid')
lines = []


for f in tqdm.tqdm(files):
    py,dur = get_pys_dur(f)
    name = f.split('/')[-1].split('.Te')[0]
    cn = name2cn[name]
    lines += [name+'|' + ' '.join(py)+'|'+cn+'|'+' '.join([str(d) for d in dur])]
with open('./aishell3/name_pys_hz_dur.txt','wt') as F:
    F.write('\n'.join(lines))
