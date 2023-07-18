import time
from options.train_options import TrainOptions
from options.valid_options import ValOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('debug0')
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    val_opt = ValOptions().parse()  # get training options
    val_dataset = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options
    # print('debug0')
    val_dataset_size = len(val_dataset)  # get the number of images in the dataset.
    print('The number of valid images = %d' % val_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.save_networks(opt.)
    #
    # model.netG.load_state_dict(torch.load('./checkpoints/glg_pix2pix/latest_net_G.pth', map_location='cuda'))
    # model.netD.load_state_dict(torch.load('./checkpoints/glg_pix2pix/latest_net_D.pth', map_location='cuda'))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    visualizer_val = Visualizer(val_opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    val_loss_min = np.Inf
    # print('debug1')
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        model.train()
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # print('debug2')
        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print('debug3')
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()   # only used in colorization_model(black & white image -> colorful images)
                visualizer.display_current_results(model.get_dual_visuals(), epoch, save_result)


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                # losses = model.get_current_losses()
                losses = model.get_dual_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        val_loss = 0
        model.eval()
        total_itersV = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                total_itersV += opt.batch_size
                iter_start_time = time.time()
                model.set_input(data)
                model.test2()
                # losses = model.get_val_losses()
                # val_loss += visualizer_val.get_val_losses(losses)
                losses = model.get_dual_losses()
                val_loss += losses['G_L1']

                if total_itersV % val_opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_itersV % val_opt.update_html_freq == 0
                    model.compute_visuals()  # only used in colorization_model(black & white image -> colorful images)
                    # print('val')
                    visualizer_val.display_current_results(model.get_dual_visuals(), epoch, save_result)

                if total_itersV % val_opt.print_freq == 0:    # print training losses and save logging information to the disk
                    # losses = model.get_val_losses()
                    losses = model.get_dual_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer_val.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer_val.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if val_loss <= val_loss_min and epoch > 10:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(val_loss_min, val_loss))
                print('')
                model.save_networks(epoch)
                val_loss_min = val_loss


