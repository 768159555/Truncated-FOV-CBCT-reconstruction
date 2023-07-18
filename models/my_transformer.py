# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


from .swin_transformer_unet import SwinTransformerSys



class SwinUnet(nn.Module):
    def __init__(self, img_size=224, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        # self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.swin_unet = SwinTransformerSys(img_size= 512,
                                patch_size= 4,
                                in_chans=1,
                                output_nc=1,
                                embed_dim= 16,
                                depths= [2, 2, 4, 2],
                                num_heads=[2, 4, 8, 8],       #[3, 6, 12, 24]
                                window_size= 8,
                                mlp_ratio= 4,
                                qkv_bias= True,
                                qk_scale= None,
                                drop_rate= 0.0,
                                drop_path_rate= 0.1,
                                ape= False,
                                patch_norm= True,
                                use_checkpoint= False)

    # def forward(self, x):           #image
    #     x=F.interpolate(x, size=(256,256), mode="bilinear")
    #     # if x.size()[1] == 1:
    #     #     x = x.repeat(1,3,1,1)
    #     output = self.swin_unet(x)
    #     output = F.interpolate(output, size=(512, 512), mode="bilinear")
    #     return output

    def forward(self, x):            #sino

        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        output = self.swin_unet(x)

        return output

class SwinUnet2(nn.Module):
    def __init__(self, img_size=224, zero_head=False, vis=False):
        super(SwinUnet2, self).__init__()
        # self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.swin_unet2 = SwinTransformerSys(img_size=512,
                                            patch_size=4,
                                            in_chans=2,
                                            output_nc=1,
                                            embed_dim=16,
                                            depths=[2, 2, 4, 2],
                                            num_heads=[2, 4, 8, 8],  # [3, 6, 12, 24]
                                            window_size=8,
                                            mlp_ratio=4,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    # def forward(self, x):           #image
    #     x=F.interpolate(x, size=(256,256), mode="bilinear")
    #     # if x.size()[1] == 1:
    #     #     x = x.repeat(1,3,1,1)
    #     output = self.swin_unet(x)
    #     output = F.interpolate(output, size=(512, 512), mode="bilinear")
    #     return output

    def forward(self, x):  # image domain

        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        output = self.swin_unet2(x)

        return output
