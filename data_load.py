#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:14:26 2022

@author: surajb
"""

import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import random
from torchvision import transforms,utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

class MarineBenthicDataset(Dataset):
    """seabed dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        
        image_filepath = self.root_dir[idx]
        image_name = os.path.basename(image_filepath)
        
        image = mpimg.imread(image_filepath)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
       # key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        #key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# transforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}

class NormalizeNew(object):
    """This is a new class to Convert a color image to grayscale and normalize the color range to [0,1].
    This is defined separately from the regular Normalize class above. 
    The difference is in how the keypoints are normalized
    to lie within a custom range. This is based on Bjartens implementation
    https://github.com/Bjarten/computer-vision-ND/blob/master/project_1_facial_keypoints/data_load.py
    """        

    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        
        # image = np.copy(image)
        # name = np.copy(key_pts)

        # # convert image to grayscale
        # image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image=  image/255.0
        
        # Taken from Bjartens implemetation
        #scale keypoints to be centered around 0 with a range of [-2, 2]
        #key_pts_copy = (key_pts_copy - image.shape[0]/2)/(image.shape[0]/4)


        return {'image': image, 'name': name}
    
    
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}
    
    
    
class RescaleCustom(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, name = sample['image'],sample['name']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        

        return {'image': img, 'name': name}    


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
    
    
    
class RandomCropCustom(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, name = sample['image'], sample['name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

       

        return {'image': image, 'name': name}    
    
    
class AngleRot(object):
    """Rotate image and keypoint annotations to a given angle.

    
    """

    def __init__(self, angle):
       # assert isinstance(output_size, (int, tuple))
        self.angle = angle

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        img = TF.rotate(image,self.angle)
        kyPts = TF.rotate(key_pts,self.angle)

        return {'image': img, 'keypoints': kyPts}
    
class RandomRotate(object):
    """Rotate image in sample by an angle - Taken from Bjartens implementation"""
    #rotation was previously set to 30 (default)
    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        rows = image.shape[0]
        cols = image.shape[1]
       # generation of rotation matrix is changed from Bjartens implementation. Here angle is randomly picked from within the      specified range
        
        M = cv2.getRotationMatrix2D((rows/2,cols/2),random.choice(np.array(range(-self.rotation,self.rotation,1))),1)
        
        image_copy = cv2.warpAffine(image_copy,M,(cols,rows))
                
        
        key_pts_copy = key_pts_copy.reshape((1,136))
        new_keypoints = np.zeros(136)
        
        for i in range(68):
            coord_idx = 2*i
            old_coord = key_pts_copy[0][coord_idx:coord_idx+2]
            new_coord = np.matmul(M,np.append(old_coord,1))
            new_keypoints[coord_idx] += new_coord[0]
            new_keypoints[coord_idx+1] += new_coord[1]
        
        new_keypoints = new_keypoints.reshape((68,2))
        
        return {'image': image_copy, 'keypoints': new_keypoints}
    
    
class RandomRotateCustom(object):
    """Rotate image in sample by an angle - Taken from Bjartens implementation"""
    #rotation was previously set to 30 (default)
    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        
        image_copy = np.copy(image)
        
        
        rows = image.shape[0]
        cols = image.shape[1]
       # generation of rotation matrix is changed from Bjartens implementation. Here angle is randomly picked from within the      specified range
        
        M = cv2.getRotationMatrix2D((rows/2,cols/2),random.choice(np.array(range(-self.rotation,self.rotation,1))),1)
        
        image_copy = cv2.warpAffine(image_copy,M,(cols,rows))
                
                
        return {'image': image_copy, 'name': name}    
    
    
    
    
class ColorJitter(object):
    """ColorJitter image in sample"""
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,)
        
        image_copy = np.copy(image)
        
        key_pts_copy = np.copy(key_pts)

        image_copy = color_jitter(Image.fromarray(image_copy)) 
       
        image_copy = np.array(image_copy)
        
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
    
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

class ToTensorCustom(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, name = sample['image'], sample['name']
         
                   
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        # Applying a normalization to pixels according to this forum: https://discuss.pytorch.org/t/understanding-transform-normalize/21730/14
        # This makes each channel zero-mean and std-dev = 1
        norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # norm_transform = transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1])
        image =norm_transform(image)
        return {'image': image,'name': name}
    
    
# class ToTensorCustom(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, name = sample['image'], sample['name']
         
                   
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
                
#         return {'image': torch.from_numpy(image),'name': name}
    
    
    
class RandomHorizontalFlip(object):
    """Random horizontal flip of image in sample"""
    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        
        #image_copy = np.copy(image)
        

        if random.choice([0, 1]) <= 0.5:
            # horizontally flip image
           # image = np.fliplr(image) 
            transforms.RandomHorizontalFlip()(image)
        return {'image': image, 'name': name}

class RandomVerticalFlip(object):
    """Random horizontal flip of image in sample"""
    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        
        
        

        if random.choice([0, 1]) <= 0.5:
            # horizontally flip image
            transforms.RandomVerticalFlip()(image) 
        return {'image': image, 'name': name}
    
    
class ColorJitter(object):
    """ColorJitter image in sample"""
    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,)
        
        #image_copy = np.copy(image)
        
        if random.choice([0, 1]) <= 0.5:

            image = color_jitter(image) 
       
        #image_copy = np.array(image_copy)
        
        return {'image': image, 'name': name}
    
def visualize_output_loader(data_loader,batch_plot_size=10):
    plt.figure(figsize=(20,8))
    plt.suptitle('Transformed images')
    sample = next(iter(data_loader))
    # iterate through the test dataset
    
    # get sample data: images and names
    images = sample['image']
    name = sample['name']
    
    batch_size = images.shape[0]
    
    for i in range(batch_plot_size):
        idx = random.randint(0,batch_plot_size)
        ax = plt.subplot(batch_plot_size/5, 5, i+1)
        ImgPlot = images[idx]
        # ImgPlot = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[(1/0.229), (1/0.224), (1/0.225)])(ImgPlot)
        ImgPlot = ImgPlot.detach().numpy()
        ImgPlot = np.transpose(ImgPlot, (1, 2, 0)) 
        ImgPlotName = name[idx]
        plt.imshow(ImgPlot)
        plt.title(ImgPlotName,fontsize = 8)
        # plt.axis('off')

    plt.show()
    
def visualize_raw_data_loader(data_loader,batch_plot_size=10):
    plt.figure(figsize=(20,8))
    plt.suptitle('Raw images')
    sample = next(iter(data_loader))
    # iterate through the test dataset
    
    # get sample data: images and names
    images = sample['image']
    name = sample['name']
    
    batch_size = images.shape[0]
    
    for i in range(batch_plot_size):
        idx = random.randint(0,batch_plot_size)
        ax = plt.subplot(batch_plot_size/5, 5, i+1)
        # ImgPlot = images[idx]
        # ImgPlot = ImgPlot.numpy()
        # ImgPlot = np.transpose(ImgPlot, (1, 2, 0)) 
        ImgPlotName = name[idx]
        plt.imshow(images[idx])
        plt.title(ImgPlotName,fontsize = 8)
        # plt.axis('off')

    plt.show()    
    
    
    

def encoder_sample_output(vae,data_loader):
    
    # iterate through the test dataset
    for i, sample in enumerate(data_loader):
        
        # get sample data: images and names
        images = sample['image']
        name = sample['name']

        

        # forward pass to get net output-original
        #encode_vector = vae(images)

        x_recon, latent_mu, latent_logvar = vae(images)
        
        # break after first image is tested
        if i == 0:
            return images,name,x_recon,latent_mu,latent_logvar
        
def visualize_vae_output(orig_image,recon_image):
    
    # Input arguments
    # orig_image - original image (torch tensor)
    # name       - original image name
    # recond_image- reconstructed image(vae output)  (torch tensor)
    # batch_plot_size - number of (original,reconstructed) image-pairs to plot
    
    # Grab batch size
    
    batch_size = orig_image.shape[0]
    
    batch_size  =4
    unnorm_transform = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    
    # orig_plot = unnorm_transform(orig_image)
    # recon_plot = unnorm_transform(recon_image)
    
    orig_plot = unnorm_transform(orig_image)
    recon_plot = unnorm_transform(recon_image)
    
    orig_plot = orig_plot.detach().numpy()
    # orig_plot = orig_image.numpy()
    orig_plot = np.transpose(orig_plot,(2,3,1,0))
    
    recon_plot = recon_plot.detach().numpy()
    # recon_plot = recon_image.numpy()
    recon_plot = np.transpose(recon_plot,(2,3,1,0))
    
    # 
    # recon_plot = (recon_plot[:,:,:,:]+0.485)*0.229 
    # 
    '''
    plt.figure(figsize=(20,10))
    plt.figure
    plt.suptitle('Reconstructed images')
    '''
    fig,axs = plt.subplots(2,batch_size,figsize=(15,5))
    plt.suptitle('Reconstructed images')
    
    #loop through rows
    for i in range(batch_size):
        # axs[0,i].imshow(((orig_plot[:,:,:,i]+np.array([0.485,0.456,0.406]))*np.array([0.229,0.224,0.225])))
        # axs[1,i].imshow(((recon_plot[:,:,:,i]+np.array([0.485,0.456,0.406]))*np.array([0.229,0.224,0.225])))
        
        axs[0,i].imshow(orig_plot[:,:,:,i])
        axs[1,i].imshow(recon_plot[:,:,:,i])
    
    
    plt.show()
    return fig
    
def visualize_vae_output_eval(orig_image,recon_image,total_loss,recon_loss,kl_loss):
    
    # Input arguments
    # orig_image - original image (torch tensor)
    
    # recond_image- reconstructed image(vae output)  (torch tensor)
    # total loss = recon + kl 
    #reconstruction loss
    #kl divergence
    
    print(total_loss)
    
    batch_size  =1 # batch here refers to number of image-image recon pairs
    unnorm_transform = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    
    # orig_plot = unnorm_transform(orig_image)
    # recon_plot = unnorm_transform(recon_image)
    
    orig_plot = unnorm_transform(orig_image)
    recon_plot = unnorm_transform(recon_image)
    
    orig_plot = orig_plot.detach().numpy()
    # orig_plot = orig_image.numpy()
    orig_plot = np.transpose(orig_plot,(2,3,1,0))
    
    orig_plot = np.squeeze(orig_plot,axis=3)
    
    recon_plot = recon_plot.detach().numpy()
    # recon_plot = recon_image.numpy()
    recon_plot = np.transpose(recon_plot,(2,3,1,0))
    recon_plot = np.squeeze(recon_plot,axis=3)
    # 
    # recon_plot = (recon_plot[:,:,:,:]+0.485)*0.229 
    # 
    
    plt.figure(figsize=(20,10))
    '''
    plt.figure
    plt.suptitle('Reconstructed images')
    '''
    # fig,axs = plt.subplots(1,2,figsize=(10,10))
    fig,axs = plt.subplots(1,2)
    # plt.suptitle('Total loss: %f,Recon loss: %f' %total_loss.item() %recon_loss.item())
    axs[0].imshow(orig_plot)
    axs[1].imshow(recon_plot)
        
    
    
    plt.show()