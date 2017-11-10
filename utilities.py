import os
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage.measure import compare_ssim as ssim
#PSNR between imageA and imageB
def get_psnr(imageA,imageB):
    maxI = 1.0
    try:
        return 20*math.log10(maxI) - 10*math.log10(mean_squared_error(imageA,imageB))
    except:
        return 20*math.log10(maxI)

#Read images from folder path specified in 'folder'
def read_dataset(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        
        if img is not None:
            img = cv2.resize(img,(64,64))
            images.append(img)
    return images

#Display last 'n' images from object 'images'
def display_images(images,n):
    size = images.shape[0]
    plt.figure(figsize=(20, 2))
    for i in range(1,n):
        ax = plt.subplot(1, n, i)
        plt.imshow(images[size - i].reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#Noise Functions
#Salt and Pepper
def salt_and_pepper(image,factor):
    noise_factor = factor
    
    x_train_noisy = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0., 255.)
    
    return x_train_noisy,x_test_noisy

#Gaussian noise
def gaussian_noise(x_train,mean,sigma,proportion):    
    x_train_noisy = []
    for i in range(x_train.shape[0]):
        image = x_train[i]
        noise = proportion*np.random.normal(mean,sigma,x_train[i].shape)       
        x_train_noisy.append(np.clip(image + noise,0,255.))
     
    return np.array(x_train_noisy)

#Poisson Noise
def poisson_noise(x_train):
    
    x_train_noisy = skimage.util.random_noise(x_train, mode='poisson', seed=None, clip=True)    
    x_train_noisy = np.clip(x_train_noisy, 0., 255.)
    
    return x_test_noisy

#Gamma Noise
def gamma_noise(x_train,shape,scale=1.0):
    
    row,col,ch = x_train[0].shape
    
    x_train_noisy = x_train + np.random.gamma(shape,scale,x_train.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0., 255.)
    
    return x_train_noisy

#Function to save keras model
def save_model_as_json(model,fileName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(fileName + ".h5")



