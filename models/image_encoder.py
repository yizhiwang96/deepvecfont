import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math

class ImageEncoder(nn.Module):

    def __init__(self, img_size, input_nc, output_nc, ngf=64, norm_layer=nn.LayerNorm):

        super(ImageEncoder, self).__init__()
        n_downsampling = int(math.log(img_size, 2))
        ks_list = [5] * (n_downsampling - n_downsampling // 3) + [3] * (n_downsampling // 3)
        stride_list = [2] * n_downsampling
        encoder = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=7 // 2, bias=True, padding_mode='replicate'),
                   norm_layer([ngf, 2 ** n_downsampling, 2 ** n_downsampling]),
                   nn.ReLU(True)]
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2, padding_mode='replicate'),
                        norm_layer([ngf * mult * 2, 2 ** (n_downsampling-1-i), 2 ** (n_downsampling-1-i)]),
                        nn.ReLU(True)]

        self.encode = nn.Sequential(*encoder)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(ngf * (2 ** n_downsampling), 2 * 128, bias=True)
 
    def forward(self, input, bottleneck_bits):
        """Standard forward"""
        ret = self.encode(input)           
        img_feat = self.flatten(ret)
        output = {}
        output['img_feat'] = img_feat
        
        return output
