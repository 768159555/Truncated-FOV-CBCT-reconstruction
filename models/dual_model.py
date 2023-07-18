import torch
from .base_model import BaseModel
from . import networks
from torch.nn import functional as F

import odl
from odl.contrib import torch as odl_torch
from .build_geometry import initialization, build_geometry
import itertools
###############################################################################
# Helper Functions
###############################################################################
para= initialization()   #work
fp = build_geometry(para)
op_modpT = odl_torch.OperatorModule(fp.adjoint)   #adjoint : reconstructed image more than 4000 times larger
fbp=odl.tomo.fbp_op(fp)
op_fbp = odl_torch.OperatorModule(fbp)            #slow



class dualModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        # parser.set_defaults(norm='batch', netG='resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>


        self.model_names = ['G','G1']

        # self.netG = networks.define_G(2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)   #image G  Unet
        # self.netG1 = networks.define_G1(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)  # sinogram  G1  swin

        self.netG = networks.define_G2(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)          #image domain,  swin,   input 2 channels
        self.netG1 = networks.define_G1(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)            #sinogram domain,  swin, input 1 channel





        self.netD = 0


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(),self.netG1.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_C = input['C' if AtoB else 'D'].to(self.device)
        self.real_D = input['D' if AtoB else 'C'].to(self.device)

    def forward(self):         # training

        self.fake_D1 = self.netG1(self.real_C)
        self.fake_D = F.interpolate(self.fake_D1, size=(650, 1024), mode="bilinear")
        self.fake_D = torch.mul(op_fbp(self.fake_D),200)
        self.fake_B = torch.cat((self.real_A,self.fake_D), 1)
        self.fake_B = self.netG(self.fake_B)

    def forward2(self):            # Validation

        self.fake_D1 = self.netG1(self.real_C)
        self.fake_D = F.interpolate(self.fake_D1, size=(650, 1024), mode="bilinear")
        self.fake_D = torch.mul(op_fbp(self.fake_D), 200)
        self.fake_B = torch.cat((self.real_A, self.fake_D), 1)
        self.fake_B = self.netG(self.fake_B)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100



    def backward_G(self):

        self.loss_G2_L1 = self.criterionL1(self.fake_D1, self.real_D)*100
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)*100
        self.loss_G = self.loss_G2_L1 + self.loss_G_L1*2
        self.loss_G.backward(create_graph=True)



    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
