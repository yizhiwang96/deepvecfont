# data loader for training image super-resolution model
import os
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as T

torch.multiprocessing.set_sharing_strategy('file_system')

class SVGDataset(data.Dataset):
    def __init__(self, root_path, char_num = 52, transform=None, mode='train'):
        super().__init__()
        self.mode = mode
        self.pkl_path = os.path.join(root_path, self.mode, f'{mode}_all.pkl')
        # self.pkl_path = os.path.join(root_path, self.mode, f'{mode}_0001-0010.pkl')
        # self.pkl_path = os.path.join(root_path, 'test', f'test_0000-0010.pkl')
        pkl_f = open(self.pkl_path, 'rb')
        print(f"Loading {self.pkl_path} pickle file ...")
        self.all_glyphs = pickle.load(pkl_f)
        pkl_f.close()
        print(f"Finished loading")
        self.char_num = char_num
        self.trans = transform
        
    def __getitem__(self, index):
        cur_glyph = self.all_glyphs[index]
        item = {}
        item['rendered'] = torch.FloatTensor(cur_glyph['rendered']).view(self.char_num, 64, 64) / 255.
        item['rendered'] = self.trans(item['rendered'])
        item['rendered_256'] = torch.FloatTensor(cur_glyph['rendered_256']).view(self.char_num, 256, 256) / 255.
        item['rendered_256'] = self.trans(item['rendered_256'])
        return item

    def __len__(self):
        return len(self.all_glyphs)


def get_loader(root_path, char_num, batch_size, mode='train'):
    SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    #SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    dataset = SVGDataset(root_path, char_num, transform, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size)
    return dataloader

