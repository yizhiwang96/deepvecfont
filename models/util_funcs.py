import torch
import torch.nn.functional as F

def select_imgs(images_of_onefont, selected_cls, opts):
    # given selected char classes, return selected imgs
    # images_of_onefont: [bs, 52, opts.image_size, opts.image_size]
    # selected_cls: [bs, nshot]
    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.unsqueeze(3)
    selected_cls_ = selected_cls_.expand(images_of_onefont.size(0), nums, opts.image_size, opts.image_size)         
    selected_img = torch.gather(images_of_onefont, 1, selected_cls_)
    return selected_img

def select_seqs(seqs_of_onefont, selected_cls, opts):

    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.unsqueeze(3)
    selected_cls_ = selected_cls_.expand(seqs_of_onefont.size(0), nums, opts.max_seq_len, opts.seq_feature_dim)         
    selected_seqs = torch.gather(seqs_of_onefont, 1, selected_cls_)
    return selected_seqs

def select_seqlens(seqlens_of_onefont, selected_cls, opts):

    nums = selected_cls.size(1)
    selected_cls_ = selected_cls.unsqueeze(2)
    selected_cls_ = selected_cls_.expand(seqlens_of_onefont.size(0), nums, 1)         
    selected_seqlens = torch.gather(seqlens_of_onefont, 1, selected_cls_)
    return selected_seqlens

def trgcls_to_onehot(all_clss, trg_cls, opts):

    trg_char_cls = trg_cls.unsqueeze(2)
    trg_char = torch.gather(all_clss, 1, trg_char_cls)
    trg_char = trg_char.squeeze(1)
    trg_char = F.one_hot(trg_char, num_classes=opts.char_categories).squeeze(dim=1)
    return trg_char



def shift_right(x, pad_value=None):
    if pad_value is None:
        # the pad arg is move from last dim to first dim
        shifted = F.pad(x, (0, 0, 0, 0, 1, 0))[:-1, :, :]
    else:
        shifted = torch.cat([pad_value, x], axis=0)[:-1, :, :]
    return shifted


def length_form_embedding(emb):
    """Compute the length of each sequence in the batch
    Args:
        emb: [seq_len, batch, depth]
    Returns:
        a 0/1 tensor: [batch]
    """
    absed = torch.abs(emb)
    sum_last = torch.sum(absed, dim=2, keepdim=True)
    mask = sum_last != 0
    sum_except_batch = torch.sum(mask, dim=(0, 2), dtype=torch.long)
    return sum_except_batch


def lognormal(y, mean, logstd, logsqrttwopi):
    y_mean = y - mean
    # print('y_mean min', torch.min(y_mean))
    logstd_exp = logstd.exp()
    # print('logstd exp', torch.min(logstd_exp))
    y_mean_divide_exp = y_mean / logstd_exp
    # print('y-mean/logstdexp', torch.min(y_mean_divide_exp))
    return -0.5 * (y_mean_divide_exp) ** 2 - logstd - logsqrttwopi

def sequence_mask(lengths, max_len=None):
    batch_size=lengths.numel()
    max_len=max_len or lengths.max()
    return (torch.arange(0,max_len,device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1)))
