#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:44:57 2022

@author: surajb

This is to test out transforms applied to data before model training
"""

#%% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



#%% Grab paths to training images




train_data_path = '../dataset_split/train/data'
test_data_path = '../dataset_split/test/data'
val_data_path = '../dataset_split/val/data'

# Empty list to store paths to images
train_image_paths = []
test_image_paths = []
val_image_paths = []

# Grab paths to images in train folder
for data_path in glob.glob(train_data_path + '/*'):
    
    train_image_paths.append(data_path)
    
train_image_paths = list(train_image_paths)

# Grab paths to images in test folder
for data_path in glob.glob(test_data_path + '/*'):
    
    test_image_paths.append(data_path)
    
test_image_paths = list(test_image_paths)


# Grab paths to images in validation folder
for data_path in glob.glob(val_data_path + '/*'):
    
    val_image_paths.append(data_path)
    
val_image_paths = list(val_image_paths)
# THIS IS WRONG - SHOULD BE VAL IMAGE PATHS

#%% Import custom classes and define transform


# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import MarineBenthicDataset,visualize_vae_output
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import RescaleCustom

class ToTensorNew(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample
         
                   
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
        return image

train_transform =  ToTensorNew()

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
        plt.axis('off')
        
    for i in range(batch_plot_size):
        idx = random.randint(0,batch_plot_size)
        ax = plt.subplot(batch_plot_size/5, 5, i+1)
        # ImgPlot = images[idx]
        # ImgPlot = ImgPlot.numpy()
        # ImgPlot = np.transpose(ImgPlot, (1, 2, 0)) 
        ImgPlotName = name[idx]
        plt.hist(images[idx].ravel(),bins=50,density=True)
        plt.title(ImgPlotName,fontsize = 8)
        plt.axis('off')

    plt.show()    

def visualize_transformed_loader(data_loader,batch_plot_size=10):
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
        ImgPlot = ImgPlot.numpy()
        ImgPlot = np.transpose(ImgPlot, (1, 2, 0)) 
        ImgPlotName = name[idx]
        plt.imshow(ImgPlot)
        plt.title(ImgPlotName,fontsize = 8)
        plt.axis('off')

    plt.show()
    
    
def visualize_image_histogram(data_loader):
   
    
    sample = next(iter(data_loader))
    # iterate through the test dataset
    
    # get sample data: images and names
    images = sample['image']
    name = sample['name']
    
    batch_size = images.shape[0]
    
    
    idx = random.randint(0,batch_size)
    
    ImgPlot = images[idx]
    ImgPlot = ImgPlot.numpy()
    # ImgPlot = np.transpose(ImgPlot, (1, 2, 0)) 
    ImgPlotName = name[idx]
    
    
    #plot original
    plt.figure()
   
    plt.imshow(ImgPlot)
    plt.title(ImgPlotName,fontsize = 8)
    
    plt.figure()
    plt.hist(ImgPlot[:,:,0].ravel(),bins=50,density=True,color="red",alpha=0.2)
    plt.hist(ImgPlot[:,:,1].ravel(),bins=50,density=True,color="green",alpha=0.2)
    plt.hist(ImgPlot[:,:,2].ravel(),bins=50,density=True,color="blue",alpha=0.2)
    
    img_transform_tensor = train_transform(ImgPlot)
    
    img_transform_np = img_transform_tensor.numpy()
    # img_transform.transpose(1,2,0)
    img_transform_np = np.transpose(img_transform_np, (1, 2, 0)) 
    
    #plot transformed
    plt.figure()
   
    plt.imshow(img_transform_np)
    plt.title(ImgPlotName,fontsize = 8)
    
    plt.figure()
    plt.hist(img_transform_np[:,:,0].ravel(),bins=50,density=True,color="red",alpha=0.2)
    plt.hist(img_transform_np[:,:,1].ravel(),bins=50,density=True,color="green",alpha=0.2)
    plt.hist(img_transform_np[:,:,2].ravel(),bins=50,density=True,color="blue",alpha=0.2)
    
    # Unnormalize
    
    img_transform_unnorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[(1/0.229), (1/0.224), 1/0.225])(img_transform_tensor)
    img_transform_unnorm = img_transform_unnorm.numpy()
    img_transform_unnorm = np.transpose(img_transform_unnorm, (1, 2, 0)) 
    #plot unnormalized
    #plot transformed
    plt.figure()
   
    plt.imshow(img_transform_unnorm)
    plt.title(ImgPlotName,fontsize = 8)
    
    plt.figure()
    plt.hist(img_transform_unnorm[:,:,0].ravel(),bins=50,density=True,color="red",alpha=0.2)
    plt.hist(img_transform_unnorm[:,:,1].ravel(),bins=50,density=True,color="green",alpha=0.2)
    plt.hist(img_transform_unnorm[:,:,2].ravel(),bins=50,density=True,color="blue",alpha=0.2)
    
    # plt.axis('off')

    plt.show()


#%% Define function for creating dataset and dataloaders

def create_datasets(batch_size, train_image_paths = train_image_paths,train_transform=train_transform):

    #Raw dataset
    raw_dataset = MarineBenthicDataset(train_image_paths)
                    
    # create datasets for training, validation and testing. 
    train_dataset = MarineBenthicDataset(train_image_paths,transform=train_transform)

    
            
        
      
    
    
    
    # load test data in batches
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size,
                             shuffle=False,  
                             num_workers=0)
    
    raw_loader = DataLoader(raw_dataset, 
                             batch_size=batch_size,
                             shuffle=False,  
                             num_workers=0)
    
    return train_loader,raw_loader, train_dataset,raw_dataset





#%% Create datasets and dataloaders
batch_size = 32
train_loader,raw_loader, train_dataset,raw_dataset = create_datasets(batch_size=batch_size)






#plot sample images from dataloader - This is to observe that transforms are 
#correctly applied before training
      
# visualize_raw_data_loader(raw_loader)
# visualize_transformed_loader(train_loader)
visualize_image_histogram(raw_loader)

