The implementation of dual_swin in the paper "A transformer-based dual-domain network for reconstructing FOV extended cone-beam CT images from truncated sinograms in radiation therapy"


Prerequisites
Windows
python=3.6
torch=1.7.0
odl




Due to the limitation, we only provide several image slices to train. You can use your own datasets to train and test.


Our code is inspired by pytorch-CycleGAN-and-pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md) and Swin-Unet (https://github.com/HuCaoFighting/Swin-Unet)