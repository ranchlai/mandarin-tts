import jieba
import pypinyin
import numpy as np
import hparams as hp
import torch
import os
import copy

#sent = '知情人士称，特朗普已表示不会?参加:当选,总统拜登的就职典礼。'
sp_tokens = '。？！?!.;；'
#if hp.with_hanzi:
with open(os.path.join(hp.preprocessed_path,'vocab_hanzi.txt')) as F:
    cn_vocab = F.read().split('\n')
with open(os.path.join(hp.preprocessed_path,'vocab_pinyin.txt')) as F:
    py_vocab = F.read().split('\n')
in_py_voab = dict([(p,True) for p in py_vocab])
cn2idx = dict([(c,i) for i,c in enumerate(cn_vocab)])
idx2cn = dict([(i,c) for i,c in enumerate(cn_vocab)])

def split2sent(text):
    new_sub = [text]
    
    while True:
        sub = copy.deepcopy(new_sub)
        new_sub = []
        for s in sub:
            sp = False
            for t in sp_tokens:
                if t in s:
                    new_sub += s.split(t)
                    sp = True
                    break
            
            if not sp:
                new_sub += [s]
        if len(new_sub) == len(sub):
            break
    tokens = [a for a in text if a in sp_tokens]
    return new_sub,tokens
            
def correct_tone3(pys):
    
    for i in range(2,len(pys)):
        if pys[i][-1] == '3' and  pys[i-1][-1] == '3' and  pys[i-2][-1] == '3':
            pys[i-1] = pys[i-1][:-1]+'2'#change the middle one
                
                
    for i in range(1,len(pys)):
        if pys[i][-1] == '3':
            if pys[i-1][-1]=='3':
                pys[i-1] = pys[i-1][:-1]+'2'
    return pys

            
def correct_tone4(pys):
    for i in range(len(pys)-1):
        if pys[i] == 'bu4':
            if pys[i+1][-1]=='4':
                pys[i] = 'bu2'
      
            
                
    return pys
    

def replace_with_sp(pys):
    for i,p in enumerate(pys):
        if p in ',，、':
            pys[i] = 'sp1'
    return pys
def correct_tone5(pys):
    for i in range(len(pys)):
            if pys[i][-1] not in '1234':
                pys[i] += '5'
    return pys

def convert_to_py(sent):
    pys = []
    for words in list(jieba.cut(sent)):
        py = pypinyin.pinyin(words,pypinyin.TONE3)
        py = [p[0] for p in py]
        pys += py
    pys = replace_with_sp(pys)
    pys = correct_tone3(pys)
    pys = correct_tone4(pys)
    pys = correct_tone5(pys)

    #for py in all_pys:
    all_pys =['sil']+ pys+['sil']
        
    
    return ' '.join(all_pys)
    
    

def convert_cn(cn_sentence):
    cn_sentence = list(cn_sentence)
    cn_sentence = ['sp1' if c in ',，：:、' else c for c in cn_sentence]
    cn_sentence = ['sil']+cn_sentence+['sil']
    cn_array = np.array([[cn2idx[c] for c in cn_sentence]])
   
    return cn_array,cn_sentence

def convert_er2(py,cn):
    py2hz = dict([(p,h) for p,h in zip(py.split(),cn)])
    py_list = py.split()
    for i,p in enumerate(py_list):
        if p == 'er2' and py2hz[p]=='儿' and i > 1 and \
        len(py_list[i-1])>2 and  py_list[i-1][-1] in '1234':
            py_er =  py_list[i-1][:-1]+'r'+ py_list[i-1][-1]
            if in_py_voab.get(py_er,False):#must in vocab
                py_list[i-1] = py_er
                py_list[i] = 'r'
                
                
                
            
    py = ' '.join(py_list)
    return py


