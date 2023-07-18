
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
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

    if opt.eval:
        model.eval()

    out_file = 'glg_test_' + opt.model
    if not os.path.exists(out_file):
        os.mkdir(out_file)

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        img_name = img_path[0].split('\\')[-1]

        # realA = data['A'].detach().cpu().numpy()
        # np.save(os.path.join(out_file, img_name[:-4] + 'realA.npy'), realA)
        # realB = data['B'].detach().cpu().numpy()
        # np.save(os.path.join(out_file, img_name[:-4] + 'realB.npy'), realB)

        fakeB = visuals['fake_B'].cpu().numpy()
        fakeB=fakeB[0,0]*3000    #4095
        fakeB=np.round(fakeB)
        fakeB=fakeB.astype(np.int16)
        # np.save(os.path.join(out_file, img_name[:-4] + 'fakeB.npy'), fakeB)
        # np.save(os.path.join(out_file, img_name[:-4] + '.npy'), np.squeeze(fakeB))
        np.save(os.path.join('K:/4DCBCT/results/clinical/Unet', img_name), np.squeeze(fakeB))

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))


