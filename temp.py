from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import numpy as np

import numpy as np
from skimage.metrics import mean_squared_error as mse
# from skimage.measure import compare_mse as mse2
from skimage.metrics import normalized_root_mse as nrm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae

def mse2(img1, img2):
    mse = np.mean( (img1 - img2) ** 2  )
    return mse

# for patient_id in range(1,2):
patient_id = 6
i=39
path_unet = '/media/czey/Elements/0_test_validation_v2/test_unet'
path_pix2pix = '/media/czey/Elements/0_test_validation_v2/test_pix2pix'
path_cyclegan = '/media/czey/Elements/0_test_validation_v2/test_cyclegan'
path_LIMAR = '/home/czey/generative_inpainting/CTarms_201116/npytodoTPS/LIMAR'
path_NMAR = '/home/czey/generative_inpainting/CTarms_201116/npytodoTPS/NMAR'
patient_names = sorted(os.listdir(path_unet))
patient_path = os.path.join(path_unet, patient_names[patient_id])
npy_names = sorted(os.listdir(patient_path))
realA_names = [x for x in npy_names if 'realA.npy' in x]
realB_names = [x for x in npy_names if 'realB.npy' in x]
fakeB_names = [x for x in npy_names if 'fakeB.npy' in x]
# realA_names

imgs_realA=[]
imgs_realB = []
unet256_fakeBs = []
pix2pix_fakeBs = []
cyclegan_fakeBs = []
LIMAR_fakeBs = []
NMAR_fakeBs = []

for realA, realB, fakeB in zip(realA_names, realB_names, fakeB_names):
#     print(realA)
    img_realA = np.load(os.path.join(path_unet, patient_names[patient_id], realA))
    img_realB = np.load(os.path.join(path_unet, patient_names[patient_id], realB))
    unet256_fakeB = np.load(os.path.join(path_unet, patient_names[patient_id], fakeB))
    pix2pix_fakeB = np.load(os.path.join(path_pix2pix, patient_names[patient_id], fakeB))
    cyclegan_fakeB = np.load(os.path.join(path_cyclegan, patient_names[patient_id], fakeB))
    LIMAR_fakeB = np.load(os.path.join(path_LIMAR, patient_names[patient_id], fakeB[:5]+'.npy'))
    NMAR_fakeB = np.load(os.path.join(path_NMAR, patient_names[patient_id], fakeB[:5]+'.npy'))

    img_realA = (np.squeeze(img_realA,1) + 1) /2 * 4095
    img_realB = (np.squeeze(img_realB,1) + 1) /2 * 4095
    unet256_fakeB = (np.squeeze(unet256_fakeB,1) + 1) /2 * 4095
    pix2pix_fakeB = (np.squeeze(pix2pix_fakeB,1) + 1) /2 * 4095
    cyclegan_fakeB = (np.squeeze(cyclegan_fakeB,1) + 1) /2 * 4095
    LIMAR_fakeB = np.expand_dims(LIMAR_fakeB, 0)
    NMAR_fakeB = np.expand_dims(NMAR_fakeB, 0)

    imgs_realA.append(img_realA)
    imgs_realB.append(img_realB)
    unet256_fakeBs.append(unet256_fakeB)
    pix2pix_fakeBs.append(pix2pix_fakeB)
    cyclegan_fakeBs.append(cyclegan_fakeB)
    LIMAR_fakeBs.append(LIMAR_fakeB)
    NMAR_fakeBs.append(NMAR_fakeB)

imgs_realA = np.vstack(imgs_realA)
imgs_realB = np.vstack(imgs_realB)
unet256_fakeBs = np.vstack(unet256_fakeBs)
pix2pix_fakeBs = np.vstack(pix2pix_fakeBs)
cyclegan_fakeBs = np.vstack(cyclegan_fakeBs)
LIMAR_fakeBs = np.vstack(LIMAR_fakeBs)
NMAR_fakeBs = np.vstack(NMAR_fakeBs)
print("patient_id: ", patient_id)

print(psnr(imgs_realA.astype('uint16'), imgs_realB.astype('uint16')))
print('psnr of unet: ',psnr(unet256_fakeBs.astype('uint16'), imgs_realB.astype('uint16')))
print('psnr of pix2pix: ',psnr(pix2pix_fakeBs.astype('uint16'), imgs_realB.astype('uint16')))
print('psnr of cyclegan: ',psnr(cyclegan_fakeBs.astype('uint16'), imgs_realB.astype('uint16')))
print('psnr of LIMAR: ',psnr(LIMAR_fakeBs.astype('uint16'), imgs_realB.astype('uint16')))
print('psnr of NMAR: ',psnr(NMAR_fakeBs.astype('uint16'), imgs_realB.astype('uint16')))