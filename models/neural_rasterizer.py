import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import math
from models.lstm_layernorm import LayerNormLSTM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence,pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralRasterizer(nn.Module):

    def __init__(self, img_size, feature_dim, hidden_size, num_hidden_layers, ff_dropout_p, rec_dropout_p, input_nc, output_nc, ngf=64, bottleneck_bits=32, norm_layer=nn.LayerNorm, mode='train'):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(NeuralRasterizer, self).__init__()
        # seq encoder
        self.input_dim = feature_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.bottleneck_bits = bottleneck_bits
        self.unbottleneck_dim = self.hidden_size * 2
        self.ff_dropout_p = float(mode =='train') * ff_dropout_p
        self.rec_dropout_p = float(mode =='train') * rec_dropout_p
        self.lstm = LayerNormLSTM(self.input_dim, self.hidden_size, self.num_hidden_layers)
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        # image decoder
        n_upsampling = int(math.log(img_size, 2))
        ks_list = [3] * (n_upsampling // 3) + [5] * (n_upsampling - n_upsampling // 3)
        stride_list = [2] * n_upsampling
        decoder = []
        mult = 2 ** (n_upsampling)
        conv = nn.ConvTranspose2d(input_nc, int(ngf * mult / 2),
                       kernel_size=ks_list[0], stride=stride_list[0],
                       padding=ks_list[0] // 2, output_padding=stride_list[0]-1)
        decoder += [conv,norm_layer([int(ngf * mult / 2), 2, 2]),nn.ReLU(True)]
        for i in range(1, n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i)
            conv = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2, output_padding=stride_list[i]-1)
            decoder += [conv,norm_layer([int(ngf * mult / 2), 2 ** (i+1) , 2 ** (i+1)]),nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)
        if mode=='test':
            for param in self.parameters():
                param.requires_grad = False

    def init_state_input(self, sampled_bottleneck):
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbottlenecks[i](sampled_bottleneck)
            (h0, c0) = unbottleneck[:, :self.unbottleneck_dim // 2], unbottleneck[:, self.unbottleneck_dim // 2:]
            init_state_hidden.append(h0.unsqueeze(0))
            init_state_cell.append(c0.unsqueeze(0))
        init_state_hidden = torch.cat(init_state_hidden, dim=0)
        init_state_cell = torch.cat(init_state_cell, dim=0)
        init_state = {}
        init_state['hidden'] = init_state_hidden
        init_state['cell'] = init_state_cell
        return init_state

    def forward(self, trg_seq, trg_char, trg_img):
        """Standard forward"""
        output, (hidden, cell) = self.lstm(trg_seq, None)
        seq_feat = torch.cat((cell[-1,:,:],hidden[-1,:,:]),-1)
        dec_input = seq_feat
        dec_input = dec_input.view(dec_input.size(0), dec_input.size(1), 1, 1)
        dec_out = self.decode(dec_input)

        output = {}
        output['gen_imgs'] = dec_out
        output['rec_loss'] = F.l1_loss(dec_out, trg_img)

        return output