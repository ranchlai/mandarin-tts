lines = open('./aishell3/name_pys_hz_dur.txt').read().split('\n')

pys = [l.split('|')[1] for l in lines]
all_pys = []
for p in pys:
    all_pys += p.split()
py_vocab = list(set(all_pys))
py_vocab.sort()
with open('./aishell3/vocab_pinyin.txt','wt') as F:
    F.write('\n'.join(py_vocab))
    
hans = [l.split('|')[2] for l in lines]
all_hans = []
for p in hans:
    all_hans += p.split()
han_vocab = list(set(all_hans))
han_vocab.sort()
with open('./aishell3/vocab_hanzi.txt','wt') as F:
    F.write('\n'.join(han_vocab))
#han_vocab
