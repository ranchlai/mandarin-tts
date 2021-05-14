import math
import os

import numpy as np
import torch
from ipdb import set_trace
from torch.utils.data import DataLoader, Dataset

import audio as Audio
import hparams as hp
from utils import pad_1D, pad_2D, process_meta

#from text import text_to_sequence, sequence_to_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self,
                 filename_py="train.txt",
                 vocab_file_py='vocab_pinyin.txt',
                 filename_hz="train_hanzi.txt",
                 vocab_file_hz='vocab_hanzi.txt',
                 sort=False,
                 descent=False):

        self.basename, self.py_text = process_meta(
            os.path.join(hp.preprocessed_path, filename_py))
        self.sort = sort

        self.py_vocab = open(os.path.join(hp.preprocessed_path,
                                          vocab_file_py)).read().split('\n')

        #assert('pad' in self.py_vocab and 'sp1' in self.py_vocab  and 'sil' in self.py_vocab)
        _, self.py_text = process_meta(
            os.path.join(hp.preprocessed_path, filename_py))

        self.py2idx = dict([(c, i) for i, c in enumerate(self.py_vocab)])

        if hp.with_hanzi:
            self.hz_vocab = open(
                os.path.join(hp.preprocessed_path,
                             vocab_file_hz)).read().split('\n')
            #  assert('pad' in self.hz_vocab and 'sp1' in self.hz_vocab  and 'sil' in self.hz_vocab)
            _, self.hz_text = process_meta(
                os.path.join(hp.preprocessed_path, filename_hz))
            self.hz2idx = dict([(c, i) for i, c in enumerate(self.hz_vocab)])

        if sort:
            names = [
                l.split('|')[0]
                for l in open(os.path.join(hp.preprocessed_path,
                                           filename)).read().split('\n')[:-1]
            ]
            mel_len = [
                np.load(hp.preprocessed_path +
                        '/mel/baker-mel-{}.npy'.format(n)).shape[0]
                for n in names
            ]
            self.map_idx = np.argsort(mel_len)
            #i=names[map_idx[-1]]
        else:
            self.map_idx = [i for i in range(len(self.basename))]

        self.map_idx_rev = self.map_idx[::-1]

        self.descent = descent

    def __len__(self):
        return len(self.py_text)

    def __getitem__(self, idx):
        if self.descent:
            idx = self.map_idx_rev[idx]
        else:
            idx = self.map_idx[idx]
        try:
            basename = self.basename[idx]
        except:
            set_trace()
        py_array = np.array(
            [self.py2idx[c] for c in self.py_text[idx].split()])
        if hp.with_hanzi:
            hz_array = np.array(
                [self.hz2idx[c] for c in self.hz_text[idx].split()])

        else:
            hz_array = None

        mel_path = os.path.join(hp.preprocessed_path, "mel",
                                "{}-mel-{}.npy".format(hp.dataset, basename))
        mel_target = np.load(mel_path)
        D_path = os.path.join(hp.preprocessed_path, "alignment",
                              "{}-ali-{}.npy".format(hp.dataset, basename))
        D = np.load(D_path)  #*0.45937500000000003
        #f0_path = os.path.join(
        #  hp.preprocessed_path, "f0", "{}-f0-{}.npy".format(hp.dataset, basename))
        #f0 = None#np.load(f0_path)
        # energy_path = os.path.join(
        # hp.preprocessed_path, "energy", "{}-energy-{}.npy".format(hp.dataset, basename))
        #   energy = None#np.load(energy_path)

        sample = {
            "id": basename,
            "text": py_array,
            "hz_text": hz_array,
            "mel_target": mel_target,  #+6.030292,
            "D": D
            #"f0": f0,
            #"energy": energy
        }

        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]

        if hp.with_hanzi:
            hz_texts = [batch[ind]["hz_text"] for ind in cut_list]

        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        # f0s = [batch[ind]["f0"] for ind in cut_list]
        # energies = [batch[ind]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print('error:', text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        if hp.with_hanzi:
            hz_texts = pad_1D(hz_texts)
        else:
            hz_texts = None

        Ds = [d - hp.duration_mean for d in Ds]

        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        # f0s = None#pad_1D(f0s)
        # energies = None#pad_1D(energies)
        #log_Ds = np.log(Ds + hp.log_offset)

        out = {
            "id": ids,
            "text": texts,
            "hz_text": hz_texts,
            "mel_target": mel_targets,
            "D": Ds,
            "log_D": Ds,
            #"#f0": f0s,
            #"energy": energies,
            "src_len": length_text,
            "mel_len": length_mel
        }

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(index_arr[i * real_batchsize:(i + 1) *
                                          real_batchsize])
            else:
                cut_list.append(
                    np.arange(i * real_batchsize, (i + 1) * real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


# if __name__ == "__main__":
#     # Test
#     dataset = Dataset('val.txt')
#     training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
#                                  drop_last=True, num_workers=0)
#     total_step = hp.epochs * len(training_loader) * hp.batch_size

#     cnt = 0
#     for i, batchs in enumerate(training_loader):
#         for j, data_of_batch in enumerate(batchs):
#             mel_target = torch.from_numpy(
#                 data_of_batch["mel_target"]).float().to(device)
#             D = torch.from_numpy(data_of_batch["D"]).int().to(device)
#             if mel_target.shape[1] == D.sum().item():
#                 cnt += 1

#     print(cnt, len(dataset))
