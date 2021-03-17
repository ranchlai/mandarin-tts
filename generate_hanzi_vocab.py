
lines = open('./data/000001-010000.txt').read().split('\n')[0::2]
names = [l.split('\t')[0] for l in lines[:-1]]
text = [l.split('\t')[1] for l in lines[:-1]]
vocab_hanzi = set(''.join(text))
vocab_hanzi = list(vocab_hanzi)
skey = lambda h:pypinyin.pinyin(h)[0][0]
vocab_hanzi = sorted(vocab_hanzi,key=skey)
vocab_hanzi = ['pad','IY1']+vocab_hanzi[5:-15]+vocab_hanzi[-2:]+['sp1','sil']
#vocab_hanzi[-10:]

#cn_vocab
with open('./data/vocab_hanzi.txt','wt') as F:
    F.write('\n'.join(vocab_hanzi))
print('file written to ./data/vocab_hanzi.txt')
