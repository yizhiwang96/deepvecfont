import argparse
# TODO: add help for the parameters


def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # TODO: basic parameters training related
    parser.add_argument('--model_name', type=str, default='main_model', choices=['main_model', 'neural_raster', 'diff_refinement', 'image_sr'],
                        help='current model_name')
    parser.add_argument('--initializer', type=str, default='uniform_unit_scaling',
                        choices=['uniform', 'orthogonal', 'uniform_unit_scaling'],
                        help=' initializer type')
    parser.add_argument('--initializer_gain', type=float, default=1.0, help='initializer initializer gain')
    parser.add_argument('--bottleneck_bits', type=int, default=128, help='latent code number of bottleneck bits')
    parser.add_argument('--char_categories', type=int, default=52, help='number of glyphs, original is 52')
    parser.add_argument('--ref_nshot', type=int, default=4, help='reference number')    
    parser.add_argument('--in_channel', type=int, default=1, help='input image channel')
    parser.add_argument('--out_channel', type=int, default=1, help='output image channel')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # data related
    parser.add_argument('--max_seq_len', type=int, default=51, help='maximum length of sequence')
    parser.add_argument('--seq_feature_dim', type=int, default=10,
                        help='feature dim (like vocab size) of one step of sequence feature')
    parser.add_argument('--test_epoch', type=int, default=125,
                        help='the testing checkpoint')
    # experiment related
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--experiment_name', type=str, default='v1.3_lstmenc_hs512')
    parser.add_argument('--data_root', type=str, default='data/vecfont_dataset')
    parser.add_argument('--ckpt_freq', type=int, default=25, help='save checkpoint frequency of epoch')
    parser.add_argument('--sample_freq', type=int, default=200, help='sample train output of steps')
    parser.add_argument('--val_freq', type=int, default=1000, help='sample validate output of steps')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    parser.add_argument('--test_sample_times', type=int, default=20, help='the sample times when testing')
    # loss weight
    parser.add_argument('--kl_beta', type=float, default=0.01, help='latent code kl loss beta')
    parser.add_argument('--pt_c_loss_w', type=float, default=0.001, help='the weight of perceptual content loss')
    parser.add_argument('--cx_loss_w', type=float, default=0.1, help='the weight of contextual loss')
    parser.add_argument('--l1_loss_w', type=float, default=1, help='the weight of image reconstruction l1 loss')
    parser.add_argument('--mdn_loss_w', type=float, default=1.0, help='the weight of mdn loss')
    parser.add_argument('--softmax_loss_w', type=float, default=1.0, help='the weight of softmax ce loss')
    # neural rasterizer
    parser.add_argument('--use_nr', type=bool, default=False, help='whether to use neural rasterization during training')

    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')

    # LSTM related
    parser.add_argument('--hidden_size', type=int, default=512, help='lstm encoder hidden_size')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='svg decoder number of hidden layers')
    parser.add_argument('--rec_dropout', type=float, default=0.3, help='LSTM rec dropout')
    parser.add_argument('--ff_dropout', type=float, default=0.5, help='LSTM feed forward dropout')

    # MDN related
    parser.add_argument('--num_mixture', type=int, default=50, help='')
    parser.add_argument('--mix_temperature', type=float, default=0.0001, help='')
    parser.add_argument('--gauss_temperature', type=float, default=0.0001, help='')
    parser.add_argument('--dont_reduce_loss', type=bool, default=False, help='')
    
    return parser
