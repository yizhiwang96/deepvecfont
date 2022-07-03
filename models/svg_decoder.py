import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from models.lstm_layernorm import LayerNormLSTM
import models.util_funcs as util_funcs

class SVGLSTMDecoder(nn.Module):
    def __init__(self, char_categories=52,
                 bottleneck_bits=32, mode='train', max_sequence_length=51,
                 hidden_size=1024, num_hidden_layers=4, feature_dim=10, ff_dropout=0.5, rec_dropout=0.3):
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
        self.unbottlenecks = nn.ModuleList([nn.Linear(bottleneck_bits + char_categories, self.unbottleneck_dim) for _ in range(self.num_hidden_layers)])
        self.input_dim = feature_dim
        self.pre_lstm_fc = nn.Linear(self.input_dim, self.hidden_size)
        self.ff_dropout_p = float(mode =='train') * ff_dropout
        self.rec_dropout_p = float(mode =='train') * rec_dropout
        self.rnn = LayerNormLSTM(self.hidden_size, self.hidden_size, self.num_hidden_layers)
        # self.predict_fc = nn.Linear(self.hidden_size, feature_dim)

    def init_state_input(self, sampled_bottleneck, trg_char):
        sampled_bottleneck_cls = torch.cat((sampled_bottleneck, trg_char),-1)
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbottlenecks[i](sampled_bottleneck_cls)
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


class SVGMDNTop(nn.Module):
    """
    Apply the Mixture Nensity Network on the top of the LSTM ouput
    Input:
        body_output: outputs from LSTM [seq_len, batch, hidden_size]
    Output:
        The MDN output. predict mode (hard=True): [seq_len, batch, 10] feature_dim = 10
        train mode or head=False: [seq_len, batch, 4 + 6 * self.num_mix * 3]
    """
    def __init__(self, num_mixture=50, seq_len=51, hidden_size=1024, hard=False, mode='train',
                 mix_temperature=0.0001, gauss_temperature=0.0001, dont_reduce=False):
        super().__init__()
        self.num_mix = num_mixture
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.command_len = 4
        self.arg_len = 6
        self.output_channel = self.command_len + self.arg_len * self.num_mix * 3
        self.hard = hard
        self.mode = mode
        self.mix_temperature = mix_temperature
        self.gauss_temperature = gauss_temperature
        self.dont_reduce = dont_reduce
        self.fc = nn.Linear(self.hidden_size, self.output_channel, bias=True)
        self.identity = nn.Identity()

    def forward(self, decoder_output, mode='train'):
        ret = self.fc(decoder_output)
        return ret
    
    def sample(self, ret, decoder_output, mode):
        # for command
        if mode == 'train':
            # use gumbel_softmax to make it differentiable
            command = F.gumbel_softmax(ret[..., :self.command_len], tau=1, hard=True)
        else:
            command = self.identity(ret[..., :self.command_len]) / self.mix_temperature
            command_max = torch.max(command, dim=-1, keepdim=True)[0]
            command = torch.exp(command - command_max)
            command = command / torch.sum(command, dim=-1, keepdim=True)
            # sample from the given probs
            command = Categorical(probs=command).sample()
            # command = torch.argmax(command,-1)
            command = F.one_hot(command, self.command_len).to(decoder_output.device).float()
        
        # for coords(augments) Note: (Categorical + gather) is differentiable
        arguments = ret[..., self.command_len:]
        # args are [seq_len, batch, 6*3*num_mix], and get [seq_len*batch*6, 3*num_mix]
        arguments = arguments.reshape([-1, 3 * self.num_mix])
        mdn_coef = self.get_mdn_coef(arguments)
        out_logmix, out_mean, out_logstd = mdn_coef['logmix'], mdn_coef['mean'], mdn_coef['logstd']
        # these are [seq_len*batch*6, num_mix]
        # apply temp to logmix
        out_logmix = self.identity(out_logmix) / self.mix_temperature
        out_logmix_max = torch.max(out_logmix, dim=-1, keepdim=True)[0]
        out_logmix = torch.exp(out_logmix - out_logmix_max)
        out_logmix = out_logmix / torch.sum(out_logmix, dim=-1, keepdim=True)
        # out_logmix = torch.argmax(out_logmix, -1)
        out_logmix = Categorical(probs=out_logmix).sample()
        # [seq_len*batch*arg_len]
        out_logmix_tmp = out_logmix.unsqueeze(1)

        chosen_mean = torch.gather(out_mean, 1, out_logmix_tmp).squeeze(1)
        chosen_logstd = torch.gather(out_logstd, 1, out_logmix_tmp).squeeze(1)
        rand_gaussian = (torch.randn(chosen_mean.size(), device=decoder_output.device) * math.sqrt(self.gauss_temperature))
        arguments = chosen_mean + torch.exp(chosen_logstd) * rand_gaussian
        # arguments = chosen_mean
        batch_size = command.size(1)
        arguments = arguments.reshape(-1, batch_size, self.arg_len)  # [seq_len, batch, arg_len]
        # concat with the command we picked
        sampled_ret = torch.cat([command, arguments], dim=-1)

        return sampled_ret

    def get_mdn_coef(self, arguments):
        """Compute mdn coefficient, aka, split arguments to 3 chunck with size num_mix"""
        logmix, mean, logstd = torch.split(arguments, self.num_mix, dim=-1)
        logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
        mdn_coef = {}
        mdn_coef['logmix'] = logmix
        mdn_coef['mean'] = mean
        mdn_coef['logstd'] = logstd
        return mdn_coef

    def get_mdn_loss(self, logmix, mean, logstd, args_flat, batch_mask, seqlen_mask):
        """Compute MDN loss term for svg decoder model."""
        logsqrttwopi = math.log(math.sqrt(2.0 * math.pi))
        lognorm = util_funcs.lognormal(args_flat, mean, logstd, logsqrttwopi)
        v = logmix + lognorm
        v = torch.logsumexp(v, 1, keepdim=True)
        v = v.reshape([self.seq_len, -1, self.arg_len])
        v = v * batch_mask

        mdn_loss_raw = torch.sum(v, dim=2)
        mdn_loss_raw = torch.mul(mdn_loss_raw, seqlen_mask)

        if self.dont_reduce:
            return -torch.mean(mdn_loss_raw, dim=[0, 1], keepdim=True)

        return -torch.mean(mdn_loss_raw)

    def svg_loss(self, mdn_top_out, target, trg_seqlen, max_seq_len, mode='train'):
        """Compute loss for svg decoder model"""
        assert mode == 'train', "Need compute loss in train mode"
        # target is [seq_len, batch, 10]
        target_commands = target[..., :self.command_len]
        target_args = target[..., self.command_len:]

        # in train mode the mdn_top_out has size [seq_len, batch, mdn_output_channel]
        predict_commands = mdn_top_out[..., :self.command_len]
        predict_args = mdn_top_out[..., self.command_len:]
        # [seq_len, batch, 6*3*num_mix]
        predict_args = predict_args.reshape([-1, 3 * self.num_mix])
        mdn_coef = self.get_mdn_coef(predict_args)
        out_logmix, out_mean, out_logstd = mdn_coef['logmix'], mdn_coef['mean'], mdn_coef['logstd']

        # create a mask for elements to ignore on it
        masktemplate = torch.Tensor([[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [0., 0., 0., 0., 1., 1.],
                                     [1., 1., 1., 1., 1., 1.]]).to(target_commands.device)
        mask = torch.matmul(target_commands, masktemplate)
        target_args_flat = target_args.reshape([-1, 1])
        seqlen_mask = util_funcs.sequence_mask(trg_seqlen, max_seq_len)
        seqlen_mask = seqlen_mask.float().transpose(0,1) # shape: [maxseqlen, batchsize]
        mdn_loss = self.get_mdn_loss(out_logmix, out_mean, out_logstd, target_args_flat, mask, seqlen_mask)
        # print('mdn loss min', torch.min(mdn_loss))

        softmax_xent_loss = torch.sum(- target_commands * F.log_softmax(predict_commands, -1), -1)
        softmax_xent_loss = torch.mul(softmax_xent_loss, seqlen_mask)
        if self.dont_reduce:
            softmax_xent_loss = torch.mean(softmax_xent_loss, dim=[1, 2], keepdim=True)
        else:
            softmax_xent_loss = torch.mean(softmax_xent_loss)

        svg_losses = {}
        svg_losses['mdn_loss'] = mdn_loss
        svg_losses['softmax_xent_loss'] = softmax_xent_loss
        return svg_losses
