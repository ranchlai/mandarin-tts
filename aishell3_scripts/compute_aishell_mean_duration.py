import numpy as np
import hparams
lines = open('./aishell3/name_pys_hz_dur.txt').read().split('\n')
durs = []
for l in lines:
    dur = l.split('|')[-1].split()
    dur = [int(d) for d in dur]
    durs += dur
print(f'total duration:{np.sum(durs)}')
print(f'total audio len:{np.sum(durs)*hparams.hop_length/hparams.sampling_rate/3600} hours')

print(f'mean duration:{np.mean(durs)}')
