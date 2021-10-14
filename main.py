import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np
from dataloader import get_loader
from models.image_encoder import ImageEncoder
from models.image_decoder import ImageDecoder
from models.modality_fusion import ModalityFusion
from models.vgg_perceptual_loss import VGGPerceptualLoss
from models.vgg_contextual_loss import VGGContextualLoss
from models.svg_decoder import SVGLSTMDecoder, SVGMDNTop
from models.svg_encoder import SVGLSTMEncoder 
from models.neural_rasterizer import NeuralRasterizer
from models import util_funcs
from options import get_parser_main_model
from data_utils.svg_utils import render
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_main_model(opts):
    exp_dir = os.path.join("experiments", opts.experiment_name)
    sample_dir = os.path.join(exp_dir, "samples")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    logfile = open(os.path.join(log_dir, "train_loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    train_loader = get_loader(opts.data_root, opts.char_categories, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.mode)
    val_loader = get_loader(opts.data_root, opts.char_categories, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, 'test')

    img_encoder = ImageEncoder(input_nc = opts.char_categories, output_nc = 1, ngf = 16, norm_layer=nn.LayerNorm)

    img_decoder = ImageDecoder(input_nc = opts.bottleneck_bits + opts.char_categories, output_nc = 1, ngf = 16, norm_layer=nn.LayerNorm)

    vggptlossfunc = VGGPerceptualLoss()

    modality_fusion = ModalityFusion(img_feat_dim = 16 * opts.image_size, hidden_size = opts.hidden_size, 
                                     ref_nshot = opts.ref_nshot, bottleneck_bits = opts.bottleneck_bits, mode=opts.mode)

    svg_encoder = SVGLSTMEncoder(char_categories=opts.char_categories,
                                 bottleneck_bits=opts.bottleneck_bits, mode=opts.mode, max_sequence_length=opts.max_seq_len,
                                 hidden_size=opts.hidden_size, num_hidden_layers=opts.num_hidden_layers,
                                 feature_dim=opts.seq_feature_dim, ff_dropout=opts.ff_dropout, rec_dropout=opts.rec_dropout)

    svg_decoder = SVGLSTMDecoder(char_categories=opts.char_categories,
                                 bottleneck_bits=opts.bottleneck_bits, mode=opts.mode, max_sequence_length=opts.max_seq_len,
                                 hidden_size=opts.hidden_size, num_hidden_layers=opts.num_hidden_layers,
                                 feature_dim=opts.seq_feature_dim, ff_dropout=opts.ff_dropout, rec_dropout=opts.rec_dropout)
       
    mdn_top_layer = SVGMDNTop(num_mixture=opts.num_mixture, seq_len=opts.max_seq_len, hidden_size=opts.hidden_size,
                              mode=opts.mode, mix_temperature=opts.mix_temperature,
                              gauss_temperature=opts.gauss_temperature, dont_reduce=opts.dont_reduce_loss)

    neural_rasterizer = NeuralRasterizer(feature_dim=opts.seq_feature_dim, hidden_size=opts.hidden_size, num_hidden_layers=opts.num_hidden_layers, 
                                         ff_dropout_p=opts.ff_dropout, rec_dropout_p=opts.rec_dropout, input_nc = 2 * opts.hidden_size, 
                                         output_nc=1, ngf=16, bottleneck_bits=opts.bottleneck_bits, norm_layer=nn.LayerNorm, mode='test')

    neural_rasterizer_fpath = os.path.join("./experiments/dvf_neural_raster/checkpoints/neural_raster_350.nr.pth")
    neural_rasterizer.load_state_dict(torch.load(neural_rasterizer_fpath))
    neural_rasterizer.eval()

    if torch.cuda.is_available() and opts.multi_gpu:
        img_encoder = nn.DataParallel(img_encoder)
        img_decoder = nn.DataParallel(img_decoder)
        svg_encoder = nn.DataParallel(svg_encoder)
        svg_decoder = nn.DataParallel(svg_decoder)
        vggptlossfunc = nn.DataParallel(vggptlossfunc)
        mdn_top_layer = nn.DataParallel(mdn_top_layer)
        modality_fusion = nn.DataParallel(modality_fusion)
        neural_rasterizer = nn.DataParallel(neural_rasterizer)
    
    img_encoder = img_encoder.to(device)
    img_decoder = img_decoder.to(device)
    modality_fusion = modality_fusion.to(device)
    vggptlossfunc = vggptlossfunc.to(device)
    svg_encoder = svg_encoder.to(device)
    svg_decoder = svg_decoder.to(device)
    mdn_top_layer = mdn_top_layer.to(device)
    neural_rasterizer = neural_rasterizer.to(device)

    all_parameters = list(img_encoder.parameters()) + list(img_decoder.parameters()) + list(modality_fusion.parameters()) +\
                     list(svg_encoder.parameters()) + list(svg_decoder.parameters()) + list(mdn_top_layer.parameters())
    optimizer = Adam(all_parameters, lr=opts.lr, betas=(opts.beta1, opts.beta2), eps=opts.eps, weight_decay=opts.weight_decay)

    if opts.tboard:
        writer = SummaryWriter(log_dir)
    
    mean = np.load('./data/mean.npz')
    std = np.load('./data/stdev.npz')
    mean = torch.from_numpy(mean).to(device).to(torch.float32)
    std = torch.from_numpy(std).to(device).to(torch.float32)
    network_modules= [img_encoder, img_decoder, modality_fusion, vggptlossfunc, svg_encoder, svg_decoder, mdn_top_layer, neural_rasterizer]
    for epoch in range(opts.init_epoch, opts.n_epochs):
        for idx, data in enumerate(train_loader):
            # network forward for a batch of data
            img_decoder_out, vggpt_loss, kl_loss, svg_losses, trg_img, ref_img, trgsvg_nr_out, synsvg_nr_out =\
                network_forward(data, mean, std, opts, network_modules)
            if opts.use_nr:
                loss = opts.l1_loss_w * img_decoder_out['img_l1loss'] + opts.pt_c_loss_w * vggpt_loss['pt_c_loss']  + opts.kl_beta * kl_loss \
                        + opts.mdn_loss_w * svg_losses['mdn_loss'] + opts.softmax_loss_w * svg_losses['softmax_xent_loss'] + opts.l1_loss_w * synsvg_nr_out['rec_loss']
            else:
                loss = opts.l1_loss_w * img_decoder_out['img_l1loss'] + opts.pt_c_loss_w * vggpt_loss['pt_c_loss']  + opts.kl_beta * kl_loss \
                        + opts.mdn_loss_w * svg_losses['mdn_loss'] + opts.softmax_loss_w * svg_losses['softmax_xent_loss']
            output_img = img_decoder_out['gen_imgs']
            img_l1loss = img_decoder_out['img_l1loss']
            mdn_loss, softmax_xent_loss = svg_losses['mdn_loss'], svg_losses['softmax_xent_loss']
            # perform optimization
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()
            batches_done = epoch * len(train_loader) + idx + 1 

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.6f}, "
                f"img_l1_loss: {img_l1loss.item():.6f}, "
                f"kl_loss: {opts.kl_beta * kl_loss.item():.6f}, "
                f"img_pt_c_loss: {opts.pt_c_loss_w * vggpt_loss['pt_c_loss']:.6f}, "
                # f"img_pt_s_loss: {vggpt_loss['pt_s_loss']:.6f}, "
                f"mdn_loss: {mdn_loss.item():.6f}, "
                f"softmax_xent_loss: {softmax_xent_loss.item():.6f}, "
                f"synsvg_nr_recloss: {synsvg_nr_out['rec_loss'].item():.6f}"
            )
            
            if batches_done % 50 == 0:
                logfile.write(message + '\n')
                print(message)
                if opts.tboard:
                    writer.add_scalar('Loss/loss', loss.item(), batches_done)
                    writer.add_scalar('Loss/img_l1_loss', img_l1loss.item(), batches_done)
                    writer.add_scalar('Loss/img_kl_loss', opts.kl_beta * kl_loss.item(), batches_done)
                    writer.add_scalar('Loss/img_perceptual_loss', opts.pt_c_loss_w * vggpt_loss['pt_c_loss'], batches_done)
                    writer.add_scalar('Loss/cmd_softmax_loss', softmax_xent_loss.item(), batches_done)
                    writer.add_scalar('Loss/coord_mdn_loss', mdn_loss.item(), batches_done)
                    writer.add_scalar('Loss/synsvg_nr_rec_loss', synsvg_nr_out['rec_loss'].item(), batches_done)
                    writer.add_image('Images/trg_img', trg_img[0], batches_done)
                    writer.add_image('Images/trgsvg_nr_img', trgsvg_nr_out['gen_imgs'][0], batches_done)
                    writer.add_image('Images/synsvg_nr_img', synsvg_nr_out['gen_imgs'][0], batches_done)
                    writer.add_image('Images/output_img', output_img[0], batches_done)
                    '''
                    for img_idx in range(52):
                        writer.add_image('Images/src_img_' + "%02d"%img_idx, ref_img[0,img_idx:img_idx+1,:,:], batches_done)
                    '''

            if opts.sample_freq > 0 and batches_done % opts.sample_freq == 0:
                
                img_sample = torch.cat((trg_img.data, output_img.data), -2)
                #img_sample_nr = torch.cat((trg_img.data, nr_out["gen_imgs"].data), -2)
                save_file = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}.png")
                #save_file_nr = os.path.join(sample_dir, f"train_epoch_{epoch}_batch_{batches_done}.nr.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)
                #save_image(img_sample_nr, save_file_nr, nrow=8, normalize=True)        
                
            if opts.val_freq > 0 and batches_done % opts.val_freq == 0:
                val_img_l1_loss = 0.0
                val_img_pt_loss = 0.0
                val_cmd_softmax_loss = 0.0
                val_coord_mdn_loss = 0.0
                val_synsvg_nr_rec_loss = 0.0
                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        val_img_decoder_out, val_vggpt_loss, val_kl_loss, val_svg_losses, val_trg_img, val_ref_img, val_trgsvg_nr_out, val_synsvg_nr_out = network_forward(val_data, mean, std, opts, network_modules)
                        
                        val_img_l1_loss += val_img_decoder_out['img_l1loss']
                        val_img_pt_loss += val_vggpt_loss['pt_c_loss']
                        val_cmd_softmax_loss += val_svg_losses['softmax_xent_loss']
                        val_coord_mdn_loss += val_svg_losses['mdn_loss']
                        val_synsvg_nr_rec_loss += val_synsvg_nr_out['rec_loss']

                    val_img_l1_loss /= len(val_loader)
                    val_img_pt_loss /= len(val_loader)
                    val_cmd_softmax_loss /= len(val_loader) 
                    val_coord_mdn_loss /= len(val_loader)
                    val_synsvg_nr_rec_loss /= len(val_loader)

                    if opts.tboard:
                        # writer.add_scalar('VAL/loss', val_loss, batches_done)
                        writer.add_scalar('VAL/img_l1_loss', val_img_l1_loss, batches_done)
                        writer.add_scalar('VAL/img_pt_loss', val_img_pt_loss, batches_done)
                        writer.add_scalar('VAL/cmd_softmax_loss', val_cmd_softmax_loss, batches_done)
                        writer.add_scalar('VAL/coord_mdn_loss', val_coord_mdn_loss, batches_done)
                        writer.add_scalar('VAL/synsvg_nr_rec_loss', val_synsvg_nr_rec_loss, batches_done)                          
                        # writer.add_scalar('VAL/b_loss', val_b_loss, batches_done)

                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_loader)}, "
                        f"Val image l1 loss: {val_img_l1_loss: .6f}, "
                        f"Val image pt loss: {val_img_pt_loss: .6f}, "
                        f"Val cmd_softmax_loss loss: {val_cmd_softmax_loss: .6f}, "
                        f"Val coord_mdn_loss loss: {val_coord_mdn_loss: .6f}, "
                    )

                    val_logfile.write(val_msg + "\n")
                    print(val_msg)
             
        if epoch % opts.ckpt_freq == 0:
            model_file_1 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.imgenc.pth")
            model_file_2 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.imgdec.pth")
            model_file_3 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.seqenc.pth")
            model_file_4 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.seqdec.pth")
            model_file_5 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.modalfuse.pth")
            model_file_6 = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.mdntl.pth")

            if torch.cuda.is_available() and opts.multi_gpu:
                torch.save(img_encoder.module.state_dict(), model_file_1)
            else:
                torch.save(img_encoder.state_dict(), model_file_1)
                torch.save(img_decoder.state_dict(), model_file_2)
                torch.save(svg_encoder.state_dict(), model_file_3)
                torch.save(svg_decoder.state_dict(), model_file_4)
                torch.save(modality_fusion.state_dict(), model_file_5)
                torch.save(mdn_top_layer.state_dict(), model_file_6)
                
    logfile.close()
    val_logfile.close()


def network_forward(data, mean, std, opts, network_moudules):

    img_encoder, img_decoder, modality_fusion, vggptlossfunc, svg_encoder, svg_decoder, mdn_top_layer, neural_rasterizer = network_moudules

    input_image = data['rendered'].to(device) # bs, opts.char_categories, opts.image_size, opts.image_size
    input_sequence = data['sequence'].to(device) 
    input_clss = data['class'].to(device) # bs, opts.char_categories, 1
    input_seqlen = data['seq_len'].to(device) # bs, opts.char_categories 1
    
    input_sequence = (input_sequence - mean) / std
    
    # randomly choose reference classes and target classes
    if opts.ref_nshot == 1:
        ref_cls = torch.randint(0, opts.char_categories, (input_image.size(0), opts.ref_nshot)).to(device)
    else:
        ref_cls_upper = torch.randint(0, opts.char_categories // 2, (input_image.size(0), opts.ref_nshot // 2)).to(device) # bs, 1
        ref_cls_lower = torch.randint(opts.char_categories // 2, opts.char_categories, (input_image.size(0), opts.ref_nshot - opts.ref_nshot // 2)).to(device) # bs, 1
        ref_cls = torch.cat((ref_cls_upper,ref_cls_lower), -1)
    
    # the input reference images 
    trg_cls = torch.randint(0, opts.char_categories, (input_image.size(0), 1)).to(device) # bs, 1
    ref_cls_multihot = torch.zeros(input_image.size(0), opts.char_categories).to(device) # bs, 1
    for ref_id in range(0,opts.ref_nshot):
        ref_cls_multihot = torch.logical_or(ref_cls_multihot, util_funcs.trgcls_to_onehot(input_clss, ref_cls[:,ref_id:ref_id+1], opts))
    ref_cls_multihot = ref_cls_multihot.to(torch.float32)
    ref_cls_multihot = ref_cls_multihot.unsqueeze(2)
    ref_cls_multihot = ref_cls_multihot.unsqueeze(3)
    ref_cls_multihot = ref_cls_multihot.expand(input_image.size(0), opts.char_categories, opts.image_size, opts.image_size)   
    ref_img = torch.mul(input_image, ref_cls_multihot)

    # randomly select a target glyph image
    trg_img = util_funcs.select_imgs(input_image, trg_cls, opts)
    # randomly select ref vector glyphs
    ref_seq = util_funcs.select_seqs(input_sequence, ref_cls, opts) # [opts.batch_size, opts.ref_nshot, opts.max_seq_len, opts.seq_feature_dim]
    # randomly select a target vector glyph
    trg_seq = util_funcs.select_seqs(input_sequence, trg_cls, opts)
    trg_seq = trg_seq.squeeze(1)
    # the one-hot target char class
    trg_char = util_funcs.trgcls_to_onehot(input_clss, trg_cls, opts)
    # shirft target sequence
    gt_trg_seq = trg_seq.clone().detach()
    trg_seq = trg_seq.transpose(0,1)
    trg_seq_shifted = util_funcs.shift_right(trg_seq)

    # run the image encoder
    img_encoder_out = img_encoder(ref_img, opts.bottleneck_bits)
    img_feat = img_encoder_out['img_feat']
    # run the svg encoder
    ref_seq_cat = ref_seq.view(ref_seq.size(0) * ref_seq.size(1), ref_seq.size(2), ref_seq.size(3)) #  [opts.batch_size * opts.ref_nshot, opts.max_seq_len, opts.seq_feature_dim]        
    ref_seq_cat = ref_seq_cat.transpose(0,1) #  [opts.max_seq_len, opts.batch_size * opts.ref_nshot,  opts.seq_feature_dim]
    se_init_state = svg_encoder.init_state_input(torch.zeros(ref_seq_cat.size(1), opts.bottleneck_bits).to(device))
    hidden, cell = se_init_state['hidden'], se_init_state['cell']
    se_hidden_ly = torch.zeros(ref_seq_cat.size(0), ref_seq_cat.size(1), opts.hidden_size).to(device)
    se_cell_ly = torch.zeros(ref_seq_cat.size(0), ref_seq_cat.size(1), opts.hidden_size).to(device)

    ref_len = ref_seq_cat.size(0)
    for t in range(0, ref_len):
        inpt = ref_seq_cat[t]
        encoder_output = svg_encoder(inpt, hidden, cell)
        output, hidden, cell = encoder_output['output'], encoder_output['hidden'], encoder_output['cell']
        se_hidden_ly[t] = hidden[-1,:,:]
        se_cell_ly[t] = cell[-1,:,:]
    
    ref_seqlen = util_funcs.select_seqlens(input_seqlen, ref_cls, opts)
    ref_seqlen = ref_seqlen.squeeze()
    ref_seqlen = ref_seqlen.view(ref_seq_cat.size(1))
    ref_seqlen = ref_seqlen.view(1, ref_seq_cat.size(1), 1)
    ref_seqlen = ref_seqlen.expand(1, ref_seq_cat.size(1), opts.hidden_size)
    se_hidden_last = torch.gather(se_hidden_ly,0,ref_seqlen)
    se_cell_last = torch.gather(se_cell_ly,0,ref_seqlen)

    seq_feat = torch.cat((se_hidden_last.squeeze(),se_cell_last.squeeze()),-1)
    # modality fusion
    mf_output = modality_fusion(img_feat, seq_feat)
    latent_feat = mf_output['latent']
    kl_loss = mf_output['kl_loss']
    # run image decoder
    img_decoder_out = img_decoder(latent_feat, trg_char, trg_img)
    
    vggpt_loss = vggptlossfunc(img_decoder_out['gen_imgs'], trg_img)
    # run the sequence decoder
    sd_init_state = svg_decoder.init_state_input(latent_feat, trg_char)
    hidden, cell = sd_init_state['hidden'], sd_init_state['cell']
    outputs = torch.zeros(trg_seq.size(0), trg_seq.size(1), opts.hidden_size).to(device)

    trg_len = trg_seq_shifted.size(0)
    for t in range(0, trg_len):
        inpt = trg_seq_shifted[t]
        decoder_output = svg_decoder(inpt, hidden, cell)
        output, hidden, cell = decoder_output['output'], decoder_output['hidden'], decoder_output['cell']
        outputs[t] = output
    
    top_output = mdn_top_layer(outputs)
    trg_seqlen = util_funcs.select_seqlens(input_seqlen, trg_cls, opts)
    trg_seqlen = trg_seqlen.squeeze()
    svg_losses = mdn_top_layer.svg_loss(top_output, trg_seq, trg_seqlen+1, opts.max_seq_len)
    sampled_svg = mdn_top_layer.sample(top_output, outputs, opts.mode)
    
    trgsvg_nr_out = neural_rasterizer(trg_seq, trg_char, trg_img)
    synsvg_nr_out = neural_rasterizer(sampled_svg, trg_char, trg_img)
    return img_decoder_out, vggpt_loss, kl_loss, svg_losses, trg_img, ref_img, trgsvg_nr_out, synsvg_nr_out

def train(opts):
    if opts.model_name == 'main_model':
        train_main_model(opts)
    elif opts.model_name == 'others':
        train_others(opts)
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
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
