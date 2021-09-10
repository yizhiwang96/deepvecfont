import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import models.util_funcs as util_funcs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class ModalityFusion(nn.Module):
    def __init__(self, img_feat_dim=128, hidden_size=128, ref_nshot=4, bottleneck_bits=32, mode='train'):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.ref_nshot = ref_nshot
        self.hidden_size = hidden_size
        self.seq_fc = nn.Linear(ref_nshot * hidden_size * 2, hidden_size, bias=True)
        self.mode = mode
        self.fc = nn.Linear(img_feat_dim +  hidden_size, 2 * bottleneck_bits, bias=True)
        # self.fc = nn.Linear(img_feat_dim, 2 * bottleneck_bits, bias=True)
    def forward(self, img_feat, seq_feat):

        # concat + fc for modality fusion, and calcaute the mean, std, kl loss
        # img_feat: [opts.batch_size,  opts.ngf * (2 ** 6)]
        # seq_feat: [opts.batch_size * opts.ref_nshot,  opts.hidden_size * 2] (hidden and cell)

        seq_feat = seq_feat.view(img_feat.size(0), self.ref_nshot,  self.hidden_size * 2)
        seq_feat = seq_feat.view(img_feat.size(0), self.ref_nshot * self.hidden_size * 2)

        feat_cat = torch.cat((img_feat, self.seq_fc(seq_feat)),-1)
        # feat_cat = img_feat
        
        dist_param = self.fc(feat_cat)

        output = {}
        
        mu = dist_param[..., :self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits:]

        if self.mode == 'train':
            # calculate the kl loss and reparamerize latent code
            epsilon = torch.randn(*mu.size(), device=mu.device)
            z = mu + torch.exp(log_sigma / 2) * epsilon
            kl = 0.5 * torch.mean(torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma)
            output['latent'] = z
            output['kl_loss'] = kl
        else:
            output['latent'] = mu
            output['kl_loss'] = 0.0

        return output
