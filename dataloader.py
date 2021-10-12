# data loader for training main model
import os
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as T

torch.multiprocessing.set_sharing_strategy('file_system')


class SVGDataset(data.Dataset):
    def __init__(self, root_path, char_num = 52, max_seq_len=51, seq_feature_dim=10, transform=None, mode='train'):
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
        self.max_seq_len = max_seq_len
        self.feature_dim = seq_feature_dim
        self.trans = transform

    def __getitem__(self, index):
        cur_glyph = self.all_glyphs[index]
        item = {}
        item['class'] = torch.LongTensor(cur_glyph['class'])
        item['seq_len'] = torch.LongTensor(cur_glyph['seq_len'])
        item['sequence'] = torch.FloatTensor(cur_glyph['sequence']).view(self.char_num, self.max_seq_len, self.feature_dim)
        item['rendered'] = torch.FloatTensor(cur_glyph['rendered']).view(self.char_num, 64, 64) / 255.
        item['rendered'] = self.trans(item['rendered'])

        item['font_id'] = torch.FloatTensor([float(cur_glyph['binary_fp'])])
        return item

    def __len__(self):
        return len(self.all_glyphs)


def get_loader(root_path, char_num, max_seq_len, seq_feature_dim, batch_size, mode='train'):
    #SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    SetRange = T.Lambda(lambda X: 1. - X )  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    dataset = SVGDataset(root_path, char_num, max_seq_len, seq_feature_dim, transform, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size)
    return dataloader


if __name__ == '__main__':
    root_path = 'data/new_data'
    max_seq_len = 51
    seq_feature_dim = 10
    batch_size = 1
    char_num = 52

    loader = get_loader(root_path, char_num, max_seq_len, seq_feature_dim, batch_size, 'train')
    fout = open('train_id_record_old.txt','w')
    for idx, batch in enumerate(loader):
        binary_fp = batch['font_id'].numpy()[0][0]
        fout.write("%05d"%int(binary_fp) + '\n')

