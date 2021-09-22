import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from models.lstm_layernorm import LayerNormLSTM
import models.util_funcs as util_funcs


class SVGLSTMEncoder(nn.Module):
    def __init__(self, char_categories=52,
                 bottleneck_bits=32, mode='train', max_sequence_length=51, hidden_size=1024,
                 num_hidden_layers=4, feature_dim=10, ff_dropout=0.5, rec_dropout=0.3):
        super().__init__()
        self.mode = mode
        self.bottleneck_bits = bottleneck_bits
        self.char_categories = char_categories
        self.command_len = 4
        self.arg_len = 6
        assert self.command_len + self.arg_len == feature_dim
        self.ff_dropout = ff_dropout
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.unbottleneck_dim = self.hidden_size * 2
        self.unbottlenecks = nn.ModuleList([nn.Linear(bottleneck_bits, self.unbottleneck_dim) for _ in range(self.num_hidden_layers)])
        self.input_dim = feature_dim
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.ff_dropout_p = float(mode =='train') * ff_dropout
        self.rec_dropout_p = float(mode =='train') * rec_dropout
        self.rnn = LayerNormLSTM(self.hidden_size, self.hidden_size, self.num_hidden_layers)


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

    def forward(self, inpt, hidden, cell):
        if inpt.size(-1) != self.hidden_size:
            inpt = self.pre_lstm_fc(inpt)
            inpt = inpt.unsqueeze(dim=0)  
        output, (hidden, cell) = self.rnn(inpt, (hidden, cell))
        decoder_output = {}
        decoder_output['output'] = output
        decoder_output['hidden'] = hidden
        decoder_output['cell'] = cell
        return decoder_output
