import pypinyin

lines = open('./data/000001-010000.txt').read().split('\n')[0::2]
names = [l.split('\t')[0] for l in lines[:-1]]
text = [l.split('\t')[1] for l in lines[:-1]]
vocab_hanzi = set(''.join(text))
vocab_hanzi = list(vocab_hanzi)
skey = lambda h:pypinyin.pinyin(h)[0][0]
vocab_hanzi = sorted(vocab_hanzi,key=skey)
vocab_hanzi = ['pad'+'IY1']+vocab_hanzi[5:-15]+vocab_hanzi[-2:]+['sp1','sil']
#vocab_hanzi[-10:]

#cn_vocab
with open('./data/vocab_hanzi.txt','wt') as F:
    F.write('\n'.join(vocab_hanzi))
    
get_text = lambda line: ''.join([t for t in line if vocab_dict.get(t,False)])
vocab_dict = dict([(v,True) for v in vocab_hanzi])

def get_bound_idx(t):
    cn_idx = 0
    bound_idx = []
    for i,_t in enumerate(t):
        if _t == '#':
            bound_idx += [cn_idx]

        if vocab_dict.get(_t,False):
            cn_idx +=1
            
    return bound_idx
        
bound_pos = {}
for n,t in zip(names,text):
    t = t.replace('#2','#').replace('#3','#').replace('#1','#').replace('#4','#')
    bound_pos.update({n:get_bound_idx(t)})
#ls ./pure_py/synth/baker/
lines = open('./BZNSYP/000001-010000.txt').read().split('\n')[:-1]

name_cn_py = dict([])
for i in range(len(lines)//2):
    name,cn = lines[i*2].split('\t')
    
    cn = get_text(cn)
    py = lines[i*2+1].split('\t')[1]
    name_cn_py.update({name: (cn,py)})
    
import pickle
with open('bound_pos.pkl','wb') as F:
    pickle.dump(bound_pos,F)
    

def augment_cn_with_specials(cn,py):
    special_tokens = ['sil','sp1','IY1']    
    new_cn = []
    j = 0
    for i,p in enumerate(py.split()):
        if p in special_tokens:
            new_cn += [p]
        else:
            new_cn += [cn[j]]
            j += 1
    return ' '.join(new_cn)
def map_name_to_cn_py(name):
    if '-' not in name:
        return name_cn_py[name]
    else:
        cn1,py1 =name_cn_py[name.split('-')[0]]
        cn2,py2 =name_cn_py[name.split('-')[1]]
        return cn1+cn2,py1+' '+py2
        
  
    
lines = open('./data/train.txt').read().split('\n')[:-1]
print('{} lines written for train'.format(len(lines)))
train_name2py = dict([l.split('|') for l in lines])
new_lines = []
for name in train_name2py.keys():
    train_py = train_name2py[name]
    cn,py = map_name_to_cn_py(name)
    cn_augmented = augment_cn_with_specials(cn,train_py)
    new_lines += [name+'|'+cn_augmented]    
with open('./data/train_cn.txt','wt') as F:
    F.write('\n'.join(new_lines))     

lines = open('./data/val.txt').read().split('\n')[:-1]
print('{} lines written for val'.format(len(lines)))

val_name2py = dict([l.split('|') for l in lines])
new_lines = []
for name in val_name2py.keys():
    val_py = val_name2py[name]
    cn,py = map_name_to_cn_py(name)
    cn_augmented = augment_cn_with_specials(cn,val_py)
    new_lines += [name+'|'+cn_augmented]    
with open('./data/val_cn.txt','wt') as F:
    F.write('\n'.join(new_lines))  
    
