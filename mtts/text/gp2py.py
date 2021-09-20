import argparse
import copy
import os
from typing import List

import jieba
import pypinyin

SPECIAL_NOTES = '。？！?!.;；:,，:'


def read_vocab(file: os.PathLike) -> List[str]:
    with open(file) as f:
        vocab = f.read().split('\n')
        vocab = [v for v in vocab if len(v) > 0 and v != '\n']
    return vocab


class TextNormal:
    def __init__(self,
                 gp_vocab_file: os.PathLike,
                 py_vocab_file: os.PathLike,
                 add_sp1=False,
                 fix_er=False,
                 add_sil=True):
        if gp_vocab_file is not None:
            self.gp_vocab = read_vocab(gp_vocab_file)
        if py_vocab_file is not None:
            self.py_vocab = read_vocab(py_vocab_file)
            self.in_py_vocab = dict([(p, True) for p in self.py_vocab])
        self.add_sp1 = add_sp1
        self.add_sil = add_sil
        self.fix_er = fix_er

        # gp2idx = dict([(c, i) for i, c in enumerate(self.gp_vocab)])
        # idx2gp = dict([(i, c) for i, c in enumerate(self.gp_vocab)])

    def _split2sent(self, text):
        new_sub = [text]
        while True:
            sub = copy.deepcopy(new_sub)
            new_sub = []
            for s in sub:
                sp = False
                for t in SPECIAL_NOTES:
                    if t in s:
                        new_sub += s.split(t)
                        sp = True
                        break

                if not sp and len(s) > 0:
                    new_sub += [s]
            if len(new_sub) == len(sub):
                break
        tokens = [a for a in text if a in SPECIAL_NOTES]

        return new_sub, tokens

    def _correct_tone3(self, pys: List[str]) -> List[str]:
        """Fix the continuous tone3 pronunciation problem"""
        for i in range(2, len(pys)):
            if pys[i][-1] == '3' and pys[i - 1][-1] == '3' and pys[i - 2][-1] == '3':
                pys[i - 1] = pys[i - 1][:-1] + '2'  # change the middle one
        for i in range(1, len(pys)):
            if pys[i][-1] == '3':
                if pys[i - 1][-1] == '3':
                    pys[i - 1] = pys[i - 1][:-1] + '2'
        return pys

    def _correct_tone4(self, pys: List[str]) -> List[str]:
        """Fixed the problem of pronouncing 不 bu2 yao4 / bu4 neng2"""
        for i in range(len(pys) - 1):
            if pys[i] == 'bu4':
                if pys[i + 1][-1] == '4':
                    pys[i] = 'bu2'
        return pys

    def _replace_with_sp(self, pys: List[str]) -> List[str]:
        for i, p in enumerate(pys):
            if p in ',，、':
                pys[i] = 'sp1'
        return pys

    def _correct_tone5(self, pys: List[str]) -> List[str]:
        for i in range(len(pys)):
            if pys[i][-1] not in '1234':
                pys[i] += '5'
        return pys

    def gp2py(self, gp_text: str) -> List[str]:

        gp_sent_list, tokens = self._split2sent(gp_text)
        py_sent_list = []
        for sent in gp_sent_list:
            pys = []
            for words in list(jieba.cut(sent)):
                py = pypinyin.pinyin(words, pypinyin.TONE3)
                py = [p[0] for p in py]
                pys += py
            if self.add_sp1:
                pys = self._replace_with_sp(pys)
            pys = self._correct_tone3(pys)
            pys = self._correct_tone4(pys)
            pys = self._correct_tone5(pys)
            if self.add_sil:
                py_sent_list += [' '.join(['sil'] + pys + ['sil'])]
            else:
                py_sent_list += [' '.join(pys)]

        if self.add_sil:
            gp_sent_list = ['sil ' + ' '.join(list(gp)) + ' sil' for gp in gp_sent_list]
        else:
            gp_sent_list = [' '.join(list(gp)) for gp in gp_sent_list]

        if self.fix_er:
            new_py_sent_list = []
            for py, gp in zip(py_sent_list, gp_sent_list):
                py = self._convert_er2(py, gp)
                new_py_sent_list += [py]
            py_sent_list = new_py_sent_list
            print(new_py_sent_list)

        return py_sent_list, gp_sent_list

    def _convert_er2(self, py, gp):
        py2hz = dict([(p, h) for p, h in zip(py.split(), gp.split())])
        py_list = py.split()
        for i, p in enumerate(py_list):
            if (p == 'er2' and py2hz[p] == '儿' and i > 1 and len(py_list[i - 1]) > 2 and py_list[i - 1][-1] in '1234'):

                py_er = py_list[i - 1][:-1] + 'r' + py_list[i - 1][-1]

                if self.in_py_vocab.get(py_er, False):  # must in vocab
                    py_list[i - 1] = py_er
                    py_list[i] = 'r'
        py = ' '.join(py_list)
        return py


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str)
    args = parser.parse_args()
    text = args.text
    tn = TextNormal('gp.vocab', 'py.vocab', add_sp1=True, fix_er=True)
    py_list, gp_list = tn.gp2py(text)
    for py, gp in zip(py_list, gp_list):
        print(py + '|' + gp)
