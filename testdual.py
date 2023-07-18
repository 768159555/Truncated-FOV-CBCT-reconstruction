"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # print('debug1')
    # hard-code some parameters for test
    opt.num_threads = 4   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    out_file = 'glg_test_' + opt.model
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    for i, data in enumerate(dataset):


        #
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_dual_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        img_name = img_path[0].split('\\')[-1]

        # realA = data['A'].detach().cpu().numpy()
        # np.save(os.path.join(out_file, img_name[:-4] + 'realA.npy'), realA)
        # realB = data['B'].detach().cpu().numpy()
        # np.save(os.path.join(out_file, img_name[:-4] + 'realB.npy'), realB)

        fakeB = visuals['fake_B'].cpu().numpy()
        fakeB=fakeB[0,0]*4095
        fakeB=np.round(fakeB)
        fakeB=fakeB.astype(np.int16)

        # fakeB1 = visuals['fake_B1'].cpu().numpy()
        # fakeB1 = fakeB1[0, 0] * 4095
        # fakeB1 = np.round(fakeB1)
        # fakeB1 = fakeB1.astype(np.int16)

        fakeD = visuals['fake_D'].cpu().numpy()
        fakeD = fakeD[0, 0] * 4095
        fakeD[fakeD<0] = 0
        fakeD = np.round(fakeD)
        fakeD = fakeD.astype(np.int16)



        # np.save(os.path.join(out_file, img_name[:-4] + 'fakeB.npy'), fakeB)
        np.save(os.path.join(out_file,'fakeB' ,img_name[:-4] + '.npy'), np.squeeze(fakeB))
        # np.save(os.path.join(out_file, 'imgfake',img_name[:-4] + '.npy'), np.squeeze(fakeB1))
        np.save(os.path.join(out_file, 'sinofake' ,img_name[:-4] + '.npy'), np.squeeze(fakeD))


        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    #
    #
    # webpage.save()  # save the HTML

