# data loader for training image super-resolution model
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')

class SVGDataset(data.Dataset):
    def __init__(self, root_path, char_num=52, transform=None, read='dirs', img_lr=128, img_hr=256, mode='train'):
        super().__init__()
        self.img_lr = img_lr
        self.img_hr = img_hr
        self.mode = mode
        self.char_num = char_num
        self.trans = transform
        self.read = read
        if self.read == 'dirs':
            self.font_paths = []
            self.dir_path = os.path.join(root_path, self.mode)
            for root, dirs, files in os.walk(self.dir_path):
                for dir_name in dirs:
                    self.font_paths.append(os.path.join(self.dir_path, dir_name))
            self.font_paths.sort()
        else:

            self.pkl_path = os.path.join(root_path, self.mode, f'{mode}_all.pkl')
            pkl_f = open(self.pkl_path, 'rb')
            print(f"Loading {self.pkl_path} pickle file ...")
            self.all_glyphs = pickle.load(pkl_f)
            pkl_f.close()
            print(f"Finished loading")
        
    def __getitem__(self, index):
        if self.read == 'dirs':
            font_path = self.font_paths[index]
            item = {}
            item['rendered_lr'] = torch.FloatTensor(np.load(os.path.join(font_path, 'rendered_' + str(self.img_lr) + '.npy'))).view(self.char_num, self.img_lr, self.img_lr) / 255.
            item['rendered_lr'] = self.trans(item['rendered_lr'])
            item['rendered_hr'] = torch.FloatTensor(np.load(os.path.join(font_path, 'rendered_' + str(self.img_hr) + '.npy'))).view(self.char_num, self.img_hr, self.img_hr) / 255.
            item['rendered_hr'] = self.trans(item['rendered_hr'])
        else:
            cur_glyph = self.all_glyphs[index]
            item = {}
            item['rendered_lr'] = torch.FloatTensor(cur_glyph['rendered']).view(self.char_num, self.img_lr, self.img_lr) / 255.
            item['rendered_lr'] = self.trans(item['rendered'])
            item['rendered_hr'] = torch.FloatTensor(cur_glyph['rendered_256']).view(self.char_num, self.img_hr, self.img_hr) / 255.
            item['rendered_hr'] = self.trans(item['rendered_hr'])
        return item

    def __len__(self):
        if self.read == 'dirs':
            return len(self.font_paths)
        else:
            return len(self.all_fonts)

def get_loader(root_path, char_num, batch_size, read_mode, img_sl, im_sh, mode='train'):
    SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    #SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    dataset = SVGDataset(root_path, char_num, transform, read_mode, img_sl, im_sh, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size)
    return dataloader

