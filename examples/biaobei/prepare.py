import glob
import argparse
import pypinyin
import os
from mtts.utils.logging import get_logger
logger = get_logger(__file__)
def augment_cn_with_sil(py_sent, cn_sent):
    sil_loc = [i for i, p in enumerate(py_sent.split()) if p == 'sil']
    han = [h for i, h in enumerate(cn_sent.split()) if h != 'sil']

    k = 0
    final = []
    for i in range(len(han) + len(sil_loc)):
        if i in sil_loc:
            final += ['sil']
        else:
            final += [han[k]]
            k += 1
    return ' '.join(final)

def write_scp(filename,scp):
        with open(filename,'wt') as f:
            f.write('\n'.join(scp)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Biaobei data pre-processing')
    parser.add_argument('--meta_file', type=str, required=False, default='name_py_hz_dur.txt')
    parser.add_argument('--wav_folder', type=str, required=True)
    parser.add_argument('--mel_folder', type=str, required=True)
    parser.add_argument('--dst_folder', type=str, required=False, default='./train')
    parser.add_argument('--generate_vocab', type=bool, required=False, default=False)


    args = parser.parse_args()
    lines = open(args.meta_file).read().split('\n')
    lines = [l.split('|') for l in lines if len(l)>0]
    files = glob.glob(f'{args.wav_folder}/*.wav')

    wav_scp = []
    mel_scp = []

    gp_scp = []
    py_scp = []
    dur_scp = []

    spk_scp = []
    all_py = []
    all_gp = []

    for name,py,gp,dur,_,_ in lines:
        mel_file = f'{args.mel_folder}/{name}.npy'
        if not os.path.exists(mel_file):
            continue
        wav_scp += [name +' ' + f'{args.wav_folder}/{name}.wav']
        mel_scp += [name +' ' +  f'{args.mel_folder}/{name}.npy']
        py_scp += [name+' '+ py]
        gp_scp += [name+' '+ gp]
        dur_scp += [name+' '+ dur]
        all_py += py.split()
        all_gp += gp.split()
        n = len(gp.split())
        spk_idx = 0
        spk_scp += [name + ' ' + ' '.join([str(spk_idx)]*n)]
    if args.generate_vocab:
        logger.warn('Caution: The vocab generated might be different from others(e.g., pretained models)')
        pyvocab = list(set(all_py))
        gpvocab = list(set(all_gp))
        pyvocab.sort()
        gpvocab.sort()
        gpvocab.sort(key=lambda x: pypinyin.pinyin(x,0)[0][0])
        with open('py.vocab','wt') as f:
            f.write('\n'.join(pyvocab))
        with open('gp.vocab','wt') as f:
            f.write('\n'.join(gpvocab))

        os.makedirs(args.dst_folder,exist_ok=True)
        write_scp('./train/wav.scp',wav_scp)
        write_scp('./train/py.scp',py_scp)
        write_scp('./train/gp.scp',gp_scp)
        write_scp('./train/dur.scp',dur_scp)
        write_scp('./train/spk.scp',spk_scp)
        write_scp('./train/mel.scp',mel_scp)



