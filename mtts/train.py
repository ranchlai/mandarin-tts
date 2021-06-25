import argparse
import os

import numpy as np
import torch
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import BatchSampler, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from mtts.datasets.dataset import Dataset, collate_fn
from mtts.loss import FS2Loss
from mtts.models.fs2_model import FastSpeech2
from mtts.optimizer import ScheduledOptim
from mtts.utils.logging import get_logger
from mtts.utils.utils import save_image

logger = get_logger(__file__)


class AverageMeter:
    def __init__(self):
        self.mel_loss_v = 0.0
        self.posnet_loss_v = 0.0
        self.d_loss_v = 0.0
        self.total_loss_v = 0.0

        self._i = 0

    def update(self, mel_loss, posnet_loss, d_loss, total_loss):
        self.mel_loss_v = ((self.mel_loss_v * self._i) + mel_loss.item()) / (self._i + 1)
        self.posnet_loss_v = ((self.posnet_loss_v * self._i) + posnet_loss.item()) / (self._i + 1)
        self.d_loss_v = ((self.d_loss_v * self._i) + d_loss.item()) / (self._i + 1)
        self.total_loss_v = ((self.total_loss_v * self._i) + total_loss.item()) / (self._i + 1)

        self._i += 1
        return self.mel_loss_v, self.posnet_loss_v, self.d_loss_v, self.total_loss_v


def split_batch(data, i, n_split):
    n = data[1].shape[0]
    k = n // n_split
    ds = [d[:, i * k:(i + 1) * k] if j == 0 else d[i * k:(i + 1) * k] for j, d in enumerate(data)]
    return ds


def shuffle(data):
    n = data[1].shape[0]
    idx = np.random.permutation(n)
    data_shuffled = [d[:, idx] if i == 0 else d[idx] for i, d in enumerate(data)]
    return data_shuffled


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restore', type=str, default='')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    logger.info(f'using device {device}')

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    dataset = Dataset(config)

    dataloader = DataLoader(dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False,
                            num_workers=config['training']['num_workers'])

    step_per_epoch = len(dataloader) * config['training']['batch_size']

    model = FastSpeech2(config)
    model = model.to(args.device)
    #model.encoder.emb_layers.to(device) # ?

    optim_conf = config['optimizer']
    optim_class = eval(optim_conf['type'])
    logger.info(optim_conf['params'])
    optimizer = optim_class(model.parameters(), **optim_conf['params'])

    if args.restore != '':
        logger.info(f'Loading checkpoint {args.restore}')
        content = torch.load(args.restore)
        model.load_state_dict(content['model'])
        optimizer.load_state_dict(content['optimizer'])
        current_step = content['step']
        start_epoch = current_step // step_per_epoch
        logger.info(f'loaded checkpoint at step {current_step}, epoch {start_epoch}')
    else:
        current_step = 0
        start_epoch = 0
        logger.info(f'Start training from scratch,step={current_step},epoch={start_epoch}')

    lrs = np.linspace(0, optim_conf['params']['lr'], optim_conf['n_warm_up_step'])
    Scheduler = eval(config['lr_scheduler']['type'])
    lr_scheduler = Scheduler(optimizer, **config['lr_scheduler']['params'])

    loss_fn = FS2Loss().to(device)
    train_logger = SummaryWriter(config['training']['log_path'])
    val_logger = SummaryWriter(config['training']['log_path'])
    avg = AverageMeter()
    for epoch in range(start_epoch, config['training']['epochs']):

        model.train()
        for i, data in enumerate(dataloader):
            data = shuffle(data)
            max_src_len = torch.max(data[-2])
            max_mel_len = torch.max(data[-1])
            for k in range(config['training']['batch_split']):
                data_split = split_batch(data, k, config['training']['batch_split'])
                tokens, duration, mel_truth, seq_len, mel_len = data_split
                #print(mel_len)
                tokens = tokens.to(device)
                duration = duration.to(device)
                mel_truth = mel_truth.to(device)
                seq_len = seq_len.to(device)
                mel_len = mel_len.to(device)
                # if torch.max(log_D) > 50:
                #  logger.info('skipping sample')
                #  continue

                mel_truth = mel_truth - config['fbank']['mel_mean']
                duration = duration - config['duration_predictor']['duration_mean']
                output = model(tokens, seq_len, mel_len, duration, max_src_len=max_src_len, max_mel_len=max_mel_len)

                mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

                mel_loss, mel_postnet_loss, d_loss = loss_fn(d_pred, duration, mel_pred, mel_postnet, mel_truth,
                                                             ~src_mask, ~mel_mask)

                total_loss = mel_postnet_loss + d_loss + mel_loss
                ml, pl, dl, tl = avg.update(mel_loss, mel_postnet_loss, d_loss, total_loss)
                lr = optimizer.param_groups[0]['lr']
                msg = f'epoch:{epoch},step:{current_step}|{step_per_epoch},loss:{tl:.3},mel:{ml:.3},'
                msg += f'mel_postnet:{pl:.3},duration:{dl:.3},{lr:.3}'

                if current_step % config['training']['log_step'] == 0:
                    logger.info(msg)

                total_loss = total_loss / config['training']['acc_step']
                total_loss.backward()
                if current_step % config['training']['acc_step'] != 0:
                    continue

                current_step += 1
                if current_step < config['optimizer']['n_warm_up_step']:
                    lr = lrs[current_step]
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if current_step % config['training']['synth_step'] == 0:
                    mel_pred = mel_pred.detach().cpu().numpy()
                    mel_truth = mel_truth.detach().cpu().numpy()
                    saved_path = os.path.join(config['training']['log_path'], f'{current_step}.png')
                    save_image(mel_truth[0][:mel_len[0]], mel_pred[0][:mel_len[0]], saved_path)
                    np.save(saved_path + '.npy', mel_pred[0])

                if current_step % config['training']['log_step'] == 0:

                    train_logger.add_scalar('total_loss', tl, current_step)
                    train_logger.add_scalar('mel_loss', ml, current_step)
                    train_logger.add_scalar('mel_postnet_loss', pl, current_step)
                    train_logger.add_scalar('duration_loss', dl, current_step)

                if current_step % config['training']['checkpoint_step'] == 0:
                    if not os.path.exists(config['training']['checkpoint_path']):
                        os.makedirs(config['training']['checkpoint_path'])
                    ckpt_file = os.path.join(config['training']['checkpoint_path'],
                                             'checkpoint_{}.pth.tar'.format(current_step))
                    content = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': current_step}

                    torch.save(content, ckpt_file)
                    logger.info(f'Saved model at step {current_step} to {ckpt_file}')
    logger.info(f"End of training for epoch {config['training']['epochs']}")
