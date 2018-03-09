import os
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.measure import compare_ssim
from scipy.stats import pearsonr
from keras import backend as K


#Read images from folder path specified in 'folder'
def read_dataset(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        
        if img is not None:
            img = cv2.resize(img,(64,64))
            images.append(img)
    return images

#A new non-reference image quality metric referenced here : http://faculty.ucmerced.edu/mhyang/papers/iccv13_denoise.pdf
def non_ref_img_denoise_metric(noisy_img,denoised_img):
	mni = noisy_img - denoised_img
	ssim_n = compare_ssim(noisy_img,mni)
	ssim_p = compare_ssim(noisy_img,denoised_img)
	return pearsonr(ssim_n,ssim_p)	



