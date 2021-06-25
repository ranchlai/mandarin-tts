import torch
import torch.nn as nn


class FS2Loss(nn.Module):
    def __init__(self):
        super(FS2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, d_pred, d_truth, mel_pred, mel_postnet, mel_truth, src_mask, mel_mask):
        d_pred = d_pred.masked_select(src_mask)
        d_truth = d_truth.masked_select(src_mask)

        mel_pred = mel_pred.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_truth = mel_truth.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel_pred, mel_truth) * 0.1
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_truth)
        d_loss = self.mae_loss(d_pred, d_truth) * 0.01

        return mel_loss, mel_postnet_loss, d_loss
