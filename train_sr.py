# code for image super-resolution, reused from ***
# -*- coding: utf-8 -*

from models.imgsr.modules import TrainOptions, create_dataset, create_model
import time
import torch
import os
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset_train, dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    all_start_time = time.time()
    exp_dir = os.path.join("experiments", opt.name)
    log_dir = os.path.join(exp_dir, "logs")
    writer = SummaryWriter(log_dir)
    criterionL1 = torch.nn.L1Loss()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset_train):  # inner loop within one epoch
            step = i + epoch * len(dataset_train)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data['rendered_lr'], data['rendered_hr'])         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights              

            iter_data_time = time.time()         


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            writer.add_scalar('TRAIN/l1_loss', criterionL1(model.real_B, model.fake_B), epoch)
            l1loss_test = 0.0
            with torch.no_grad():
                for i, data in enumerate(dataset_test):
                    model.set_input(data['rendered_lr'], data['rendered_hr'])
                    model.forward()
                    l1loss_test += criterionL1(model.real_B, model.fake_B)
                l1loss_test = l1loss_test / len(dataset_test)
                print('testing l1 loss:%f' % l1loss_test)
                writer.add_image('Images/test_gen_img', model.fake_B[0], epoch)
                writer.add_image('Images/test_trg_img', model.real_B[0], epoch)
                writer.add_image('Images/test_src_img', model.real_A[0], epoch)
                writer.add_scalar('TEST/l1_loss', l1loss_test, epoch)

                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    print('Total consuming time is: %d sec' % (time.time() - all_start_time))
        
    
