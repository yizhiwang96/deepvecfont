import os
import random
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, RMSprop
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from dataloader import get_loader
from models.neural_rasterizer import NeuralRasterizer
from models.vgg_perceptual_loss import VGGPerceptualLoss
from models.vgg_contextual_loss import VGGContextualLoss
from models import util_funcs
from options import get_parser_main_model
from data_utils.svg_utils import render
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_nr_model(opts):
    exp_dir = os.path.join("experiments", opts.experiment_name)
    sample_dir = os.path.join(exp_dir, "samples")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    logfile = open(os.path.join(log_dir, "train_loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    train_loader = get_loader(opts.data_root, opts.image_size, opts.char_categories, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.read_mode, opts.mode)
    val_loader = get_loader(opts.data_root, opts.image_size, opts.char_categories, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.read_mode, 'test')

    neural_rasterizer = NeuralRasterizer(img_size=opts.image_size, feature_dim=opts.seq_feature_dim, hidden_size=opts.hidden_size, num_hidden_layers=opts.num_hidden_layers, 
                                         ff_dropout_p=opts.ff_dropout, rec_dropout_p=opts.rec_dropout, input_nc=2 * opts.hidden_size, 
                                         output_nc=1, ngf=16, bottleneck_bits=opts.bottleneck_bits, norm_layer=nn.LayerNorm, mode='train')
    
    vggcxlossfunc = VGGContextualLoss()

    if torch.cuda.is_available() and opts.multi_gpu:
        neural_rasterizer = nn.DataParallel(neural_rasterizer)
        vggcxlossfunc = nn.DataParallel(vggcxlossfunc)

    neural_rasterizer = neural_rasterizer.to(device)
    vggcxlossfunc = vggcxlossfunc.to(device)

    all_parameters = list(neural_rasterizer.parameters()) 
    optimizer = Adam(all_parameters, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    if opts.tboard:
        writer = SummaryWriter(log_dir)

    mean = np.load(os.path.join(opts.data_root, opts.mode, 'mean.npz'))
    std = np.load(os.path.join(opts.data_root, opts.mode, 'stdev.npz'))
    mean = torch.from_numpy(mean).to(device).to(torch.float32)
    std = torch.from_numpy(std).to(device).to(torch.float32)

    for epoch in range(opts.init_epoch, opts.n_epochs):

        for idx, data in enumerate(train_loader):

            input_image = data['rendered'].to(device) # bs, opts.char_categories, opts.image_size, opts.image_size
            input_sequence = data['sequence'].to(device)
            input_sequence = (input_sequence - mean) / std
            input_seqlen = data['seq_len'].to(device) # bs, opts.char_categories 1
            input_clss = data['class'].to(device) # bs, opts.char_categories, 1
            trg_cls = torch.randint(0, opts.char_categories, (input_image.size(0), 1)).to(device) # bs, 1
            # randomly select a target vector glyph
            trg_seq = util_funcs.select_seqs(input_sequence, trg_cls, opts)
            trg_seq = trg_seq.squeeze(1)
            trg_char = util_funcs.trgcls_to_onehot(input_clss, trg_cls, opts)
            # randomly select a target glyph image and svg
            trg_img = util_funcs.select_imgs(input_image, trg_cls, opts)
            gt_trg_seq = trg_seq.clone().detach()
            trg_seq = trg_seq.transpose(0,1) # seqlen, bs ,feat_dim
            # run the neural_rasterizer
            nr_out = neural_rasterizer(trg_seq, trg_char, trg_img)

            output_img = nr_out['gen_imgs']
            rec_loss = nr_out['rec_loss']
            vggcx_loss = vggcxlossfunc(nr_out['gen_imgs'], trg_img)
            loss = opts.l1_loss_w * nr_out['rec_loss'] + opts.cx_loss_w * vggcx_loss['cx_loss']

            # perform optimization
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            batches_done = epoch * len(train_loader) + idx + 1

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"img_l1_loss: {rec_loss.item():.6f}, "
                f"img_cx_loss: {opts.cx_loss_w * vggcx_loss['cx_loss']:.6f}, "
            )
            logfile.write(message + '\n')
            if batches_done % 50 == 0:
                print(message)

            if opts.tboard:
                writer.add_scalar('Loss/loss', loss.item(), batches_done)
                writer.add_scalar('Loss/img_l1_loss', opts.l1_loss_w * rec_loss.item(), batches_done)
                writer.add_scalar('Loss/img_perceptual_loss', opts.cx_loss_w * vggcx_loss['cx_loss'], batches_done)
                writer.add_image('Images/trg_img', trg_img[0], batches_done)
                writer.add_image('Images/output_img', output_img[0], batches_done)

            if opts.sample_freq > 0 and batches_done % opts.sample_freq == 0:
                img_sample = torch.cat((trg_img.data, output_img.data), -2)
                save_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)
                
                svg_target = gt_trg_seq.clone().detach()
                svg_target = svg_target * std  + mean
                for i, one_gt_seq in enumerate(svg_target):
                    cur_svg_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}_no_{i}_svg.svg")
                    if i == 0:
                        gt_svg = render(one_gt_seq.cpu().numpy())
                        with open(cur_svg_file, 'a') as f:
                            f.write(gt_svg+'\n')
                        break
                
            if opts.val_freq > 0 and batches_done % opts.val_freq == 0:
                val_img_l1_loss = 0.0
                val_img_pt_loss = 0.0

                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        
                        val_input_image = val_data['rendered'].to(device)
                        val_input_clss = val_data['class'].to(device)
                        val_input_sequence = val_data['sequence'].to(device)
                        val_input_sequence = (val_input_sequence - mean) / std
                        val_input_seqlen = val_data['seq_len'].to(device)
                        val_trg_cls = torch.randint(0, opts.char_categories, (val_input_image.size(0), 1)).to(device) # bs, 1
                        val_trg_img = util_funcs.select_imgs(val_input_image, val_trg_cls, opts)
                        val_trg_seq = util_funcs.select_seqs(val_input_sequence, val_trg_cls, opts)
                        val_trg_seq = val_trg_seq.squeeze(1)
                        val_trg_seq = val_trg_seq.transpose(0, 1) # seqlen, bs ,feat_dim
                        val_trg_char = util_funcs.trgcls_to_onehot(val_input_clss, val_trg_cls, opts)
                        # run the image encoder-decoder
                        val_nr_out = neural_rasterizer(val_trg_seq, val_trg_char, val_trg_img)
                        val_output_image = val_nr_out['gen_imgs']
                        val_rec_loss = val_nr_out['rec_loss']
                        val_vggcx_loss = vggcxlossfunc(val_output_image, val_trg_img)
                        val_img_l1_loss += val_rec_loss.item()
                        val_img_pt_loss += val_vggcx_loss['cx_loss']
                    
                    val_img_l1_loss /= len(val_loader)
                    val_img_pt_loss /= len(val_loader)

                    val_img_sample = torch.cat((val_trg_img.data, val_output_image.data), -2)
                    val_save_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}.png")
                    save_image(val_img_sample, val_save_file, nrow=8, normalize=True)

                    val_svg_target = val_trg_seq.clone().detach()
                    val_svg_target = val_svg_target * std  + mean
                    #cur_svg_file = os.path.join(res_dir, f"val_epoch_{epoch}_batch_{val_idx}_svg.svg")
                    for i, one_gt_seq in enumerate(val_svg_target):
                        cur_svg_file = os.path.join(sample_dir, f"val_epoch_{epoch}_batch_{batches_done}_no_{i}_svg.svg")
                        if i == 0:
                            gt_svg = render(one_gt_seq.cpu().numpy())
                            with open(cur_svg_file, 'a') as f:
                                f.write(gt_svg+'\n')
                            break

                    if opts.tboard:
                        writer.add_scalar('VAL/img_l1_loss', val_img_l1_loss, batches_done)
                        writer.add_scalar('VAL/img_pt_loss', val_img_pt_loss, batches_done)

                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"Val image l1 loss: {val_img_l1_loss: .6f}, "
                        f"Val image pt loss: {val_img_pt_loss: .6f}, "
                    )

                    val_logfile.write(val_msg + "\n")
                    print(val_msg)
             

        if epoch % opts.ckpt_freq == 0:
            model_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.nr.pth")
            if torch.cuda.is_available() and opts.multi_gpu:
                torch.save(neural_rasterizer.module.state_dict(), model_fpath)
            else:
                torch.save(neural_rasterizer.state_dict(), model_fpath)
                
    logfile.close()
    val_logfile.close()

def train(opts):
    if opts.model_name == 'neural_raster':
        train_nr_model(opts)
    elif opts.model_name == 'others':
        train_others(opts)
    else:
        raise NotImplementedError

def test(opts):
    if opts.model_name == 'neural_raster':
        train_nr_model(opts)
    elif opts.model_name == 'others':
        test_others(opts)
    else:
        raise NotImplementedError

def main():
    opts = get_parser_main_model().parse_args()
    opts.experiment_name = opts.experiment_name + '_' + opts.model_name
    os.makedirs("experiments", exist_ok=True)
    debug = True
    if opts.mode == 'train':
        # Create directories
        experiment_dir = os.path.join("experiments", opts.experiment_name)
        os.makedirs(experiment_dir, exist_ok=debug)  # False to prevent multiple train run by mistake
        os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train(opts)
    elif opts.mode == 'test':
        print(f"Testing on experiment {opts.experiment_name}...")
        test(opts)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
