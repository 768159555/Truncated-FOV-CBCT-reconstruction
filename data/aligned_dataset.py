import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import cv2
import odl
from odl.contrib import tomo
from models.build_geometry import initialization, build_geometry


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # get the image directory
        self.dir_C = os.path.join(opt.sinoroot, opt.phase + 'A')  # get the sino directory
        self.dir_D = os.path.join(opt.sinoroot, opt.phase + 'B')  # get the sino directory


        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))  # get image paths
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))  # get image paths

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        D_path = self.D_paths[index]


        A = np.load(A_path)
        B = np.load(B_path)
        A = np.squeeze(A)
        B = np.squeeze(B)

        C = np.load(C_path)
        D = np.load(D_path)
        C = np.squeeze(C)
        D = np.squeeze(D)


        AA = np.zeros((512,512))    #image
        AA[128:384, 128:384] = A      #256
        A=AA

        C = cv2.resize(C, (264, 512))  #sinogram mask 256
        D = cv2.resize(D, (512, 512))
        CC = np.zeros((512, 512))
        CC[:, 124:388] = C   #256
        C = CC


        A[A > 4095] = 4095       #image
        B[B > 4095] = 4095
        A[A < 0] = 0
        B[B < 0] = 0
        A = A / 4095
        B = B / 4095

        simax = 1000000  # sinogram   4095*256=1048320
        C[C > simax] = simax
        D[D > simax] = simax
        C[C < 0] = 0
        D[D < 0] = 0
        C = C / simax
        D = D / simax
        C = np.float32(C)
        D = np.float32(D)



        A = np.float32(np.expand_dims(A, axis=0))
        B = np.float32(np.expand_dims(B, axis=0))
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        C = np.float32(np.expand_dims(C, axis=0))
        D = np.float32(np.expand_dims(D, axis=0))
        C = torch.from_numpy(C)
        D = torch.from_numpy(D)

        # return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        return {'A': A, 'B': B, 'C': C, 'D': D, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'D_paths': D_path}
        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
