import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from dataloader import get_loader
from models.image_encoder import ImageEncoder
from models.image_decoder import ImageDecoder
from models.modality_fusion import ModalityFusion
from models.vgg_perceptual_loss import VGGPerceptualLoss
from models.svg_decoder import SVGLSTMDecoder, SVGMDNTop
from models.svg_encoder import SVGLSTMEncoder
from models import util_funcs
from options import get_parser_main_model
from data_utils.svg_utils import render
from models.imgsr.modules import TrainOptions, create_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_main_model(opts):
    exp_dir = os.path.join("experiments", opts.experiment_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    res_dir = os.path.join(exp_dir, "results")
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    test_loader = get_loader(opts.data_root, opts.image_size, opts.char_categories, opts.max_seq_len, opts.seq_feature_dim, opts.batch_size, opts.read_mode, 'test')

    img_encoder = ImageEncoder(img_size=opts.image_size, input_nc=opts.char_categories, output_nc=1, ngf=16, norm_layer=nn.LayerNorm)

    img_decoder = ImageDecoder(img_size=opts.image_size, input_nc=opts.bottleneck_bits + opts.char_categories, output_nc=1, ngf=16, norm_layer=nn.LayerNorm)
    
    vggptlossfunc = VGGPerceptualLoss()

    modality_fusion = ModalityFusion(img_feat_dim=16 * opts.image_size, hidden_size=opts.hidden_size, ref_nshot=opts.ref_nshot, bottleneck_bits=opts.bottleneck_bits, mode=opts.mode)

    svg_encoder = SVGLSTMEncoder(char_categories=opts.char_categories,
                                 bottleneck_bits=opts.bottleneck_bits, mode=opts.mode, max_sequence_length=opts.max_seq_len,
                                 hidden_size=opts.hidden_size,
                                 num_hidden_layers=opts.num_hidden_layers,
                                 feature_dim=opts.seq_feature_dim, ff_dropout=opts.ff_dropout, rec_dropout=opts.rec_dropout)

    svg_decoder = SVGLSTMDecoder(char_categories=opts.char_categories,
                                 bottleneck_bits=opts.bottleneck_bits, mode=opts.mode, max_sequence_length=opts.max_seq_len,
                                 hidden_size=opts.hidden_size,
                                 num_hidden_layers=opts.num_hidden_layers,
                                 feature_dim=opts.seq_feature_dim, ff_dropout=opts.ff_dropout, rec_dropout=opts.rec_dropout)
    
    mdn_top_layer = SVGMDNTop(num_mixture=opts.num_mixture, seq_len=opts.max_seq_len, hidden_size=opts.hidden_size,
                              mode=opts.mode, mix_temperature=opts.mix_temperature,
                              gauss_temperature=opts.gauss_temperature, dont_reduce=opts.dont_reduce_loss)


    # load parameters
    epoch = opts.test_epoch
    img_encoder_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.imgenc.pth")
    img_encoder.load_state_dict(torch.load(img_encoder_fpath))
    img_encoder.eval()

    img_decoder_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.imgdec.pth")
    img_decoder.load_state_dict(torch.load(img_decoder_fpath))
    img_decoder.eval()

    svg_encoder_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.seqenc.pth")
    svg_encoder.load_state_dict(torch.load(svg_encoder_fpath))
    svg_encoder.eval()

    svg_decoder_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.seqdec.pth")
    svg_decoder.load_state_dict(torch.load(svg_decoder_fpath))
    svg_decoder.eval()

    modality_fusion_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.modalfuse.pth")
    modality_fusion.load_state_dict(torch.load(modality_fusion_fpath))
    modality_fusion.eval()

    mdn_top_layer_fpath = os.path.join(ckpt_dir, f"{opts.model_name}_{epoch}.mdntl.pth")
    mdn_top_layer.load_state_dict(torch.load(mdn_top_layer_fpath))
    mdn_top_layer.eval()

    # to device
    img_encoder = img_encoder.to(device)
    img_decoder = img_decoder.to(device)
    modality_fusion = modality_fusion.to(device)
    vggptlossfunc = vggptlossfunc.to(device)
    svg_encoder = svg_encoder.to(device)
    svg_decoder = svg_decoder.to(device)
    mdn_top_layer = mdn_top_layer.to(device)


    val_img_l1_loss = 0.0
    val_img_pt_loss = 0.0
    mean = np.load(os.path.join(opts.data_root, 'train', 'mean.npz'))
    std = np.load(os.path.join(opts.data_root, 'train', 'stdev.npz'))

    mean = torch.from_numpy(mean).to(device).to(torch.float32)
    std = torch.from_numpy(std).to(device).to(torch.float32)
    network_modules= [img_encoder, img_decoder, modality_fusion, vggptlossfunc, svg_encoder, svg_decoder, mdn_top_layer]

    # image super resolution model
    imgsr_opt = TrainOptions().parse()   # get training options
    imgsr_opt.isTrain = False
    imgsr_opt.batch_size = 1
    imgsr_opt.phase = 'test'
    imgsr_model = create_model(imgsr_opt)      # create a model given opt.model and other options
    imgsr_model.setup(imgsr_opt)               # regular setup: load and print networks; create schedulers
    
    with torch.no_grad():
        for test_idx, test_data in enumerate(test_loader):
            print("testing font %04d ..."%test_idx)
            img_decoder_out, vggpt_loss, kl_loss, svg_losses, trg_img, ref_img, gt_trg_seq, sampled_svg_list =\
                network_forward(test_data, mean, std, opts, network_modules)
            
            output_img = img_decoder_out["gen_imgs"]
            imgsr_model.set_test_input(1.0 - 2 * output_img)
            #imgsr_model.set_test_input(1.0 - 2 * torch.tile(output_img,(1,3,1,1))) # from [0,1] to [-1,1] and change fore/background color
            with torch.no_grad():
                imgsr_model.forward()

            output_img_hr = imgsr_model.fake_B 
            savedir_idx = os.path.join(res_dir, "%04d"%test_idx)

            if not os.path.exists(savedir_idx):
                os.mkdir(savedir_idx)
                os.mkdir(os.path.join(savedir_idx, "imgs_" + str(opts.image_size)))
                os.mkdir(os.path.join(savedir_idx, "imgs_" + str(opts.image_size_sr)))
                os.mkdir(os.path.join(savedir_idx, "svgs"))

            img_sample_merge = torch.cat((trg_img.data, output_img.data), -2)
            save_file_merge = os.path.join(savedir_idx, "imgs_" + str(opts.image_size), f"merge_" + str(opts.image_size) + ".png")
            save_image(img_sample_merge, save_file_merge, nrow=8, normalize=True)    

            for char_idx in range(opts.char_categories):
                img_gt = (1.0 - trg_img[char_idx,...]).data
                save_file_gt = os.path.join(savedir_idx, "imgs_" + str(opts.image_size), f"{char_idx:02d}_gt.png")
                save_image(img_gt, save_file_gt, normalize=True)

                img_sample = (1.0 - output_img[char_idx,...]).data
                save_file = os.path.join(savedir_idx,"imgs_" + str(opts.image_size), f"{char_idx:02d}_" + str(opts.image_size) + ".png")
                #save_image(img_sample, save_file, nrow=8, normalize=True)
                save_image(img_sample, save_file, normalize=True)
                
                img_sample_hr = output_img_hr[char_idx,...].data
                save_file_hr = os.path.join(savedir_idx,"imgs_" + str(opts.image_size_sr), f"{char_idx:02d}_" + str(opts.image_size_sr) + ".png")
                #save_image(img_sample_hr, save_file_hr, nrow=8, normalize=True)
                save_image(img_sample_hr, save_file_hr, normalize=True) 
            
            # save the generated svgs and gt svgs
            syn_svg_merge_f = open(os.path.join(os.path.join(savedir_idx,"svgs"), f"syn_merge.html"), 'w')
            for sample_id, sampled_svg in enumerate(sampled_svg_list):

                svg_dec_out = sampled_svg.clone().detach()
                svg_dec_out = svg_dec_out.transpose(0,1)
                svg_dec_out = svg_dec_out * std  + mean
                
                for i, one_seq in enumerate(svg_dec_out):
                    syn_svg_outfile = os.path.join(os.path.join(savedir_idx,"svgs"), f"syn_{i:02d}_{sample_id:02d}.svg")
                    syn_svg_f = open(syn_svg_outfile, 'w')
                    try:
                        svg = render(one_seq.cpu().numpy())
                        syn_svg_f.write(svg)
                        syn_svg_merge_f.write(svg)
                        if i > 0 and i % 13 == 12:
                            syn_svg_merge_f.write('<br>')
                        
                    except:
                        continue
                    syn_svg_f.close()
                
                syn_svg_f.close()

            svg_target = gt_trg_seq.clone().detach()
            svg_target = svg_target * std  + mean
            for i, one_gt_seq in enumerate(svg_target):
                gt_svg_outfile = os.path.join(os.path.join(savedir_idx,"svgs"), f"gt_{i:02d}.svg")
                gt_svg_f = open(gt_svg_outfile, 'w')
                gt_svg = render(one_gt_seq.cpu().numpy())
                gt_svg_f.write(gt_svg)
                syn_svg_merge_f.write(gt_svg)
                gt_svg_f.close()
                if i > 0 and i % 13 == 12:
                    syn_svg_merge_f.write('<br>')
            gt_svg_f.close()
            syn_svg_merge_f.close()
                                            
        
        val_img_l1_loss /= len(test_loader)
        val_img_pt_loss /= len(test_loader)

        val_msg = (
            f"Epoch: {epoch}/{opts.n_epochs}, "
            # f"Val loss: {val_loss: .6f}, "
            f"Val image l1 loss: {val_img_l1_loss: .6f}, "
            f"Val image pt loss: {val_img_pt_loss: .6f}, "
            #f"Val kl loss: {val_b_loss: .6f}"
        )
        print(val_msg)

def network_forward(data, mean, std, opts, network_moudules):

    img_encoder, img_decoder, modality_fusion, vggptlossfunc, svg_encoder, svg_decoder, mdn_top_layer = network_moudules

    input_image = data['rendered'].to(device) # bs, opts.char_categories, opts.image_size, opts.image_size
    input_sequence = data['sequence'].to(device) 
    input_clss = data['class'].to(device) # bs, opts.char_categories, 1
    input_seqlen = data['seq_len'].to(device) # bs, opts.char_categories 1
    
    input_sequence = (input_sequence - mean) / std
    
    # randomly choose reference classes and target classes
    if opts.ref_nshot == 1:
        ref_cls = torch.randint(0, opts.char_categories, (input_image.size(0), opts.ref_nshot)).to(device)
    else:
        ref_cls_upper = torch.tensor([[0,1]]).to(device) # A B
        ref_cls_lower = torch.tensor([[26,27]]).to(device) # a, b
        #ref_cls_upper = torch.randint(0, opts.char_categories // 2, (input_image.size(0), opts.ref_nshot // 2)).to(device) # bs, 1
        #ref_cls_lower = torch.randint(opts.char_categories // 2, opts.char_categories, (input_image.size(0), opts.ref_nshot - opts.ref_nshot // 2)).to(device) # bs, 1
        ref_cls = torch.cat((ref_cls_upper,ref_cls_lower), -1)
    
    # the input reference images 
    trg_cls = torch.randint(0, opts.char_categories, (input_image.size(0), 1)).to(device) # bs, 1
    trg_cls = torch.arange(0, opts.char_categories).to(device) # bs, 1
    trg_cls = trg_cls.view(opts.char_categories, 1)

    ref_cls_multihot = torch.zeros(input_image.size(0), opts.char_categories).to(device) # bs, 1
    for ref_id in range(0,opts.ref_nshot):
        ref_cls_multihot = torch.logical_or(ref_cls_multihot, util_funcs.trgcls_to_onehot(input_clss, ref_cls[:,ref_id:ref_id+1], opts))
    ref_cls_multihot = ref_cls_multihot.to(torch.float32)
    ref_cls_multihot = ref_cls_multihot.unsqueeze(2)
    ref_cls_multihot = ref_cls_multihot.unsqueeze(3)
    ref_cls_multihot = ref_cls_multihot.expand(input_image.size(0), opts.char_categories, opts.image_size, opts.image_size)   
    ref_img = torch.mul(input_image, ref_cls_multihot)

    # randomly select a target glyph image
    trg_img = util_funcs.select_imgs(input_image.repeat(opts.char_categories,1,1,1), trg_cls, opts)
    # randomly select ref vector glyphs
    ref_seq = util_funcs.select_seqs(input_sequence, ref_cls, opts) # [opts.batch_size, opts.ref_nshot, opts.max_seq_len, opts.seq_feature_dim]
    # randomly select a target vector glyph
    trg_seq = util_funcs.select_seqs(input_sequence.repeat(opts.char_categories,1,1,1), trg_cls, opts)
    trg_seq = trg_seq.squeeze(1)
    # the one-hot target char class
    trg_char = util_funcs.trgcls_to_onehot(input_clss.repeat(opts.char_categories,1,1), trg_cls, opts)
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
    latent_feat = latent_feat.repeat(opts.char_categories, 1)
    img_decoder_out = img_decoder(latent_feat, trg_char, trg_img)
    
    vggpt_loss = vggptlossfunc(img_decoder_out['gen_imgs'], trg_img)


    trg_len = trg_seq_shifted.size(0)

    sampled_svg_list = []

    for tst in range(opts.test_sample_times):
        # run the sequence decoder
        sd_init_state = svg_decoder.init_state_input(latent_feat, trg_char)
        hidden, cell = sd_init_state['hidden'], sd_init_state['cell']
        outputs = torch.zeros(trg_seq.size(0), trg_seq.size(1), opts.hidden_size).to(device)
        hidden_self, cell_self = hidden, cell
        # outputs_self = torch.zeros(trg_seq.size(0), trg_seq.size(1), opts.hidden_size).to(device)
        sampled_svg = torch.zeros(trg_seq.size(0), trg_seq.size(1), opts.seq_feature_dim).to(device)
        for t in range(0, trg_len):
            # self sample results
            if t == 0:
                inpt_self = torch.zeros(trg_seq.size(1), opts.seq_feature_dim).to(device)
            else:
                inpt_self = sampled_svg[t-1]
            decoder_output_self = svg_decoder(inpt_self, hidden_self, cell_self)
            output_self, hidden_self, cell_self =  decoder_output_self['output'], decoder_output_self['hidden'], decoder_output_self['cell']
            top_output_self = mdn_top_layer(output_self)
            sampled_step = mdn_top_layer.sample(top_output_self, output_self, opts.mode)
            sampled_svg[t] = sampled_step 

        sampled_svg_list.append(sampled_svg)

        top_output = mdn_top_layer(outputs)
        trg_seqlen = util_funcs.select_seqlens(input_seqlen.repeat(opts.char_categories,1,1), trg_cls, opts)
        trg_seqlen = trg_seqlen.squeeze()

        svg_losses = mdn_top_layer.svg_loss(top_output, trg_seq, trg_seqlen+1, opts.max_seq_len)


    return img_decoder_out, vggpt_loss, kl_loss, svg_losses, trg_img, ref_img, gt_trg_seq, sampled_svg_list


def test(opts):
    if opts.model_name == 'main_model':
        test_main_model(opts)
    elif opts.model_name == 'others':
        test_main_model(opts)
    else:
        raise NotImplementedError


def main():
    opts = get_parser_main_model().parse_args()
    opts.experiment_name = opts.experiment_name + '_' + opts.model_name
    os.makedirs("experiments", exist_ok=True)
    debug = True
    if opts.mode == 'test':
        print(f"Testing on experiment {opts.experiment_name}...")
        test(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
