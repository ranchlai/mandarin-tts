from ipdb import set_trace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio
from download_utils import download_checkpoint,download_waveglow

    
parser = argparse.ArgumentParser()
parser.add_argument('--restore_step', type=int, default=410000)
args = parser.parse_args()
#def main(args):
#torch.manual_seed(0)
# Get dataset


import pickle
with open('./aishell_mean_emb.pkl','rb') as F:
    emb = pickle.load(F)
    
if hp.with_hanzi:
    dataset = Dataset(filename_py="train_pinyin.txt",vocab_file_py = 'vocab_pinyin.txt',
                 filename_hz = "train_hanzi.txt",
                 vocab_file_hz = 'vocab_hanzi.txt')
    py_vocab_size = len(dataset.py_vocab)
    hz_vocab_size = len(dataset.hz_vocab)
    
    
else:
    dataset = Dataset(filename_py="train_pinyin.txt",vocab_file_py = 'vocab_pinyin.txt',
                 filename_hz = None,
                 vocab_file_hz = None)
    py_vocab_size = len(dataset.py_vocab)
    hz_vocab_size = None
# Get device
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# Get dataset

loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                    collate_fn=dataset.collate_fn, drop_last=False, num_workers=8)
# Define model
model = FastSpeech2(py_vocab_size,hz_vocab_size).to(device)
num_param = utils.get_param_num(model)

# Optimizer and loss
optimizer = torch.optim.Adam(
    model.parameters(), lr = hp.start_lr,betas=hp.betas, eps=hp.eps, weight_decay=0)
scheduled_optim = ScheduledOptim(
    optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
Loss = FastSpeech2Loss().to(device)
print("Optimizer and Loss Function Defined.")
# Load checkpoint if exists
checkpoint_path = os.path.join(hp.checkpoint_path)
try:
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
    
    #temp = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    #model.load_state_dict(temp.module.state_dict())
    #del temp
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("\n---Model Restored at Step {}---\n".format(args.restore_step))
except:
    print("\n---Start New Training---\n")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
# Load vocoder
#if hp.vocoder == 'melgan':
#waveglow = utils.get_waveglow()
hp.vocoder = 'waveglow' #force to use waveglow
print('loading waveglow...')
waveglow = download_waveglow(device)



# Init logger
log_path = hp.log_path
if not os.path.exists(log_path):
    os.makedirs(log_path)
    os.makedirs(os.path.join(log_path, 'train'))
    os.makedirs(os.path.join(log_path, 'validation'))
train_logger = SummaryWriter(os.path.join(log_path, 'train'))
val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

# Init synthesis directory
synth_path = hp.synth_path
if not os.path.exists(synth_path):
    os.makedirs(synth_path)

# Define Some Information
Time = np.array([])
Start = time.perf_counter()
# Training
best_val_loss = 1000
best_t_l = 1000


d_l,  m_l, m_p_l = evaluate(
    model, 0)

    
model = model.train()
for epoch in range(0,hp.epochs):
   
    #dataset.map_idx = local_shuffle(dataset.map_idx)
    print('optimizer lr:',optimizer.param_groups[0]['lr'])
    # Get Training Loader
    total_step = hp.epochs * len(loader) * hp.batch_size
    bar = tqdm.tqdm(total=len(dataset)//hp.batch_size)
    t_l = 0.0
    m_l = 0.0
    m_p_l =0.0
    d_l = 0.0
    K = 0
    for i, batchs in enumerate(loader):
        
        for j, data_of_batch in enumerate(batchs):
            bar.update(1)
            start_time = time.perf_counter()

            current_step = i*hp.batch_size + j + args.restore_step + \
                epoch*len(loader)*hp.batch_size + 1

            # Get Data
            text = torch.from_numpy(
                data_of_batch["text"]).long().to(device)
          
            
            if hp.with_hanzi:
                hz_text = torch.from_numpy(
                data_of_batch["hz_text"]).long().to(device)
            else:
                hz_text = None
                
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            
            #mel_target -= hp.mel_mean
            
            D = torch.from_numpy(data_of_batch["D"]).long().to(device)
            log_D = torch.from_numpy(
                data_of_batch["log_D"]).float().to(device)

            if torch.max(log_D) > 400:
                
                print(f'skipping sample because log_D max is {torch.max(log_D)}')
                continue

            src_len = torch.from_numpy(
                data_of_batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(
                data_of_batch["mel_len"]).long().to(device)
            max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
           # set_trace()
            # Forward
            
            spk_emb = [np.expand_dims(emb[sid[:7]],[0,1]) for sid in data_of_batch['id']]
            spk_emb = np.concatenate(spk_emb,0)
            spk_emb = torch.tensor(spk_emb).cuda()
            output = model(
                src_seq=text, 
                speaker_emb = spk_emb,
                src_len=src_len, 
                hz_seq=hz_text,
                mel_len=mel_len,
                d_target=D, 
                max_src_len=max_src_len, 
                max_mel_len=max_mel_len)
            
            mel_output, mel_postnet_output,log_duration_output, src_mask, mel_mask, _ = output
            # Cal Loss
 
            mel_loss, mel_postnet_loss, d_loss = Loss(
                log_duration_output, log_D,   
                mel_output, mel_postnet_output, mel_target-hp.mel_mean, ~src_mask, ~mel_mask)
            
            
            total_loss =  mel_postnet_loss + d_loss# + mel_loss
            
            # Logger
            t_l = (total_loss.item() + K*t_l)/(1+K)
            m_l = (mel_loss.item() + K*m_l)/(1+K)
            m_p_l = (mel_postnet_loss.item() + K*m_p_l)/(1+K)
            d_l = (d_loss.item() + K*d_l)/(1+K)
            K +=1
            
            lr =  optimizer.param_groups[0]['lr'] 
            msg = 'total:{:.3},mel:{:.3},mel_postnet:{:.3},duration:{:.3},{:.3}'.format(t_l,m_l,m_p_l,d_l,lr)
            bar.set_description_str(msg)
           
            
            # Backward
            total_loss = total_loss / hp.acc_steps
            total_loss.backward()
            if current_step % hp.acc_steps != 0:
                continue

            # Clipping gradients to avoid gradient explosion
           # nn.utils.clip_grad_norm_(
             #   model.parameters(), hp.grad_clip_thresh)
            if current_step == hp.n_warm_up_step:
                optimizer.param_groups[0]['lr']=1e-4
            if current_step < hp.n_warm_up_step:
            # Update weights
                scheduled_optim.step_and_update_lr()            #optimizer.step()
                scheduled_optim.zero_grad()
            else:
                optimizer.step()            #optimizer.step()
                optimizer.zero_grad()

            # Print
            if current_step % hp.log_step == 0:
                Now = time.perf_counter()

                str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                    epoch+1, hp.epochs, current_step, total_step)
                print(str1)
                
                str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    t_l, m_l, m_p_l, d_l)
                print(str2)
                
                
                str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))
                
                print(str3)
                
                
                
                train_logger.add_scalar(
                    'Loss/total_loss', t_l, current_step)
                train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                train_logger.add_scalar(
                    'Loss/mel_postnet_loss', m_p_l, current_step)
                train_logger.add_scalar(
                    'Loss/duration_loss', d_l, current_step)
               

            if current_step % hp.save_step == 0:
                torch.save({'lr':optimizer.param_groups[0]['lr'], 'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                print("save model at step {} ...".format(current_step))

            if current_step % hp.synth_step == 0:
                length = mel_len[0].item()
                mel_target_torch = mel_target[0, :length].detach(
                ).unsqueeze(0).transpose(1, 2)
                mel_target = mel_target[0, :length].detach(
                ).cpu().transpose(0, 1)
                mel_torch = mel_output[0, :length].detach(
                ).unsqueeze(0).transpose(1, 2)
                mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                mel_postnet_torch = mel_postnet_output[0, :length].detach(
                ).unsqueeze(0).transpose(1, 2)
                mel_postnet = mel_postnet_output[0, :length].detach(
                ).cpu().transpose(0, 1)
                
                
               
               # Audio.tools.inv_mel_spec(mel, os.path.join(
                   # synth_path, "step_{}_griffin_lim.wav".format(current_step)))
               # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
                    #synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))

                if hp.vocoder == 'melgan':
                    utils.melgan_infer(mel_torch, melgan, os.path.join(
                        hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                    utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(
                        hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                    utils.melgan_infer(mel_target_torch, melgan, os.path.join(
                        hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
                elif hp.vocoder == 'waveglow':
                   # utils.waveglow_infer(mel_torch, waveglow, os.path.join(
                       # hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                    utils.waveglow_infer(mel_postnet_torch+hp.mel_mean, waveglow, os.path.join(
                        hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                    utils.waveglow_infer(mel_target_torch, waveglow, os.path.join(
                        hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))


                utils.plot_data([mel_postnet.numpy()+hp.mel_mean, mel_target.numpy()],
                                ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))

          
            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)

    if t_l >= best_t_l*0.99 and (epoch+1) % 2==0:
        optimizer.param_groups[0]['lr'] *= 0.98
        print('train loss not decreasing, using new lr',optimizer.param_groups[0]['lr'])
        
        
    if t_l < best_t_l:
        best_t_l = t_l
        print('best training loss found:',best_t_l)
        
        
    d_l,  m_l, m_p_l = evaluate(
        model, current_step)
    t_l = d_l + m_l + m_p_l

    #if t_l >= best_val_loss*0.95:
       # optimizer.param_groups[0]['lr'] *= 0.95
       # print('val loss not decreasing, using new lr',optimizer.param_groups[0]['lr'])

    if t_l < best_val_loss:
        best_val_loss = t_l
        best_name =  os.path.join(checkpoint_path, 'val_best_{}_{:.3}.pth.tar'.format(current_step,best_val_loss))
        torch.save( model.state_dict(),best_name)
        print('saving best model to ',best_name)

    else:
        print('model not improving,current loss {},best loss {}'.format(t_l,best_val_loss))
    #else:


    val_logger.add_scalar(
        'Loss/total_loss', t_l, current_step)
    val_logger.add_scalar(
        'Loss/mel_loss', m_l, current_step)
    val_logger.add_scalar(
        'Loss/mel_postnet_loss', m_p_l, current_step)
    val_logger.add_scalar(
        'Loss/duration_loss', d_l, current_step)


    model.train()

                
