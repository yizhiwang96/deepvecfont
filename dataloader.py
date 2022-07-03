# data loader for training main model
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')

class SVGDataset(data.Dataset):
    def __init__(self, root_path, img_size=128, char_num = 52, max_seq_len=51, seq_feature_dim=10, transform=None, read='dirs', mode='train'):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.char_num = char_num
        self.max_seq_len = max_seq_len
        self.feature_dim = seq_feature_dim
        self.trans = transform
        self.read = read
        if self.read == 'dirs':
            self.font_paths = []
            self.dir_path = os.path.join(root_path, self.mode)
            for root, dirs, files in os.walk(self.dir_path):
                for dir_name in dirs:
                    self.font_paths.append(os.path.join(self.dir_path, dir_name))
            self.font_paths.sort()
            print(f"Finished loading {mode} paths")
        else:
            self.pkl_path = os.path.join(root_path, self.mode, f'{mode}_all.pkl')
            pkl_f = open(self.pkl_path, 'rb')
            print(f"Loading {self.pkl_path} pickle file ...")
            self.all_fonts = pickle.load(pkl_f)
            pkl_f.close()
            print(f"Finished loading pkls")

    def __getitem__(self, index):
        if self.read == 'dirs':
            font_path = self.font_paths[index]
            item = {}
            item['class'] = torch.LongTensor(np.load(os.path.join(font_path, 'class.npy')))
            item['seq_len'] = torch.LongTensor(np.load(os.path.join(font_path, 'seq_len.npy')))
            item['sequence'] = torch.FloatTensor(np.load(os.path.join(font_path, 'sequence.npy'))).view(self.char_num, self.max_seq_len, self.feature_dim)
            item['rendered'] = torch.FloatTensor(np.load(os.path.join(font_path, 'rendered_' + str(self.img_size) + '.npy'))).view(self.char_num, self.img_size, self.img_size) / 255.
            item['rendered'] = self.trans(item['rendered'])
            item['font_id'] = torch.FloatTensor(np.load(os.path.join(font_path, 'font_id.npy')).astype(np.float32))
        else:
            cur_glyph = self.all_fonts[index]
            item = {}
            item['class'] = torch.LongTensor(cur_glyph['class'])
            item['seq_len'] = torch.LongTensor(cur_glyph['seq_len'])
            item['sequence'] = torch.FloatTensor(cur_glyph['sequence']).view(self.char_num, self.max_seq_len, self.feature_dim)
            item['rendered'] = torch.FloatTensor(cur_glyph['rendered']).view(self.char_num, self.img_size, self.img_size) / 255.
            item['rendered'] = self.trans(item['rendered'])
            item['font_id'] = torch.FloatTensor([float(cur_glyph['binary_fp'])])
        return item

    def __len__(self):
        if self.read == 'dirs':
            return len(self.font_paths)
        else:
            return len(self.all_fonts)


def get_loader(root_path, img_size, char_num, max_seq_len, seq_feature_dim, batch_size, read_mode, mode='train'):
    #SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    dataset = SVGDataset(root_path, img_size, char_num, max_seq_len, seq_feature_dim, transform, read_mode, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size, drop_last=True)
    return dataloader

if __name__ == '__main__':
    root_path = 'data/new_data'
    max_seq_len = 51
    seq_feature_dim = 10
    batch_size = 1
    char_num = 52

    loader = get_loader(root_path, char_num, max_seq_len, seq_feature_dim, batch_size, 'dirs', 'train')
    fout = open('train_id_record_old.txt','w')
    for idx, batch in enumerate(loader):
        binary_fp = batch['font_id'].numpy()[0][0]
        fout.write("%05d"%int(binary_fp) + '\n')

