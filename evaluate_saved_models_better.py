#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:52:35 2022

@author: surajb
"""
#%% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import yaml
from data_load import create_datasets,RescaleCustom, RandomCropCustom, NormalizeNew, ToTensorCustom, RandomRotateCustom, RandomHorizontalFlip,RandomVerticalFlip,NewColorJitter
from torchvision import transforms, utils
from data_load import MarineBenthicDataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from numpy.random import RandomState
import seaborn as sns


#%%  Define helper functions

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def vae_loss(recon_x, x, mu, logvar,variational_beta):
    
    
    # Mean squared error loss - by default, the loss is averaged over number of elements,
    # alternatives for reduction are 'none' or 'sum'
    recon_loss = torch.nn.MSELoss(reduction='sum') (recon_x,x)
    #recon_loss  = 0.4*torch.nn.L1Loss(reduction='mean') (recon_x,x)+(1-ssim(recon_x,x))
    
    kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0) # mostly consistent with https://arxiv.org/pdf/1907.08956v1.pdf
    
        
    
    return recon_loss + variational_beta * kldivergence, recon_loss, kldivergence

# Function for evaluating reconstruction loss and extracting latent space from a dataset
def evaluate_loss_latent(dataset,config):
    # function for evaluating trained model loss and latent space
    #inputs
    # - dataset object
    # - config dictionary
    
    length_dataset = len(dataset)
    latent_size = int(config['model_params']['latent_dim'])
    
    #initialize array for storing loss
    
    total_loss_over_time = np.zeros(length_dataset)
    recon_loss_over_time = np.zeros(length_dataset)
    kl_loss_over_time = np.zeros(length_dataset)
    
    latent_mu = np.zeros((length_dataset,latent_size))
    latent_logvar = np.zeros((length_dataset,latent_size)) 
    
    model.eval()
    
    for idx, data in enumerate(dataset):
        
        # get the input images in each batch and their corresponding names
        images = data['image']
        name = data['name']
        
        #images = images.to(device)
        images = torch.unsqueeze(images,0)
        # forward pass the images through the network
        
        image_recon,latent_sample_z,mu_z,log_var_z = model(images)
        
        # Calculate loss
        total_loss,recon_loss,kl_loss = vae_loss(image_recon, images, mu_z,log_var_z,variational_beta)
        total_loss_over_time[idx] = total_loss.item()
        recon_loss_over_time[idx]=recon_loss.item()
        kl_loss_over_time[idx] = kl_loss.item()
        
        latent_logvar[idx,:] = log_var_z.detach().cpu().numpy()
        latent_mu[idx,:] = mu_z.detach().cpu().numpy()
              
    return latent_mu,recon_loss_over_time


def reduce_dimensions(features,num_components=2):
    pca = PCA(n_components=num_components,whiten=False)
    features = StandardScaler().fit_transform(features)
    features = MinMaxScaler().fit_transform(features)
        
    rng = RandomState(0)
    t_sne = manifold.TSNE(
        n_components=num_components,learning_rate=200,verbose=1,
        perplexity=40,
        n_iter=2000,
        init="pca",early_exaggeration=12.0,
        random_state=24,metric="cosine")
    

    reduced_features = t_sne.fit_transform(features)
    return reduced_features

def load_dataset_dataloader(config):
    train_data_path = config['data_params']['train_data_path']
    test_data_path = config['data_params']['test_data_path']
    val_data_path = config['data_params']['val_data_path']
    target_data_path = config['data_params']['target_data_path']
    
    # Empty list to store paths to images
    train_image_paths = []
    test_image_paths = []
    val_image_paths = []
    target_image_paths=[]
    
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
    
    
    for data_path in glob.glob(target_data_path + '/*'):
        
        target_image_paths.append(data_path)
        
    target_image_paths = list(target_image_paths)
    
    
    
    
    transform = transforms.Compose([RescaleCustom(64), ToTensorCustom()])
    
    
                    
    # create datasets for training, validation and testing. 
    test_dataset = MarineBenthicDataset(test_image_paths,transform=transform)
    
    # create new training dataset for each epoch
    train_dataset = MarineBenthicDataset(train_image_paths,transform=transform)
    
    
            
    target_dataset = MarineBenthicDataset(target_image_paths,transform=transform)
      
    
    # load training data in batches
    train_loader = DataLoader(train_dataset, 
                              batch_size=32,
                              shuffle=True, 
                              num_workers=0)
    
    
    # load test data in batches
    test_loader = DataLoader(test_dataset, 
                             batch_size=len(test_dataset),
                             shuffle=True,  
                             num_workers=0)
    
    target_loader = DataLoader(target_dataset, 
                             batch_size=len(target_dataset),
                             shuffle=True,  
                             num_workers=0)
    return train_dataset,test_dataset,target_dataset

#%% Main routine

# Create the parser
my_parser = argparse.ArgumentParser(description='train and test a model using a specified config file')

# Add the arguments
my_parser.add_argument('Experimentname',
                       metavar='experimentname',
                       type=int,
                       help='the name of the experiment to evaluate')

# Execute the parse_args() method
args = my_parser.parse_args()

experiment_name = args.Experimentname
CONFIG_PATH = str(experiment_name) + '/'
config_file_name = str(experiment_name) + '.yaml'


config = load_config(config_file_name)

# define model

latent_dims = config['model_params']['latent_dim']
variational_beta = config['model_params']['kld_weight']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.05)
        
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(p=0.05)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=0.1)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(p=0.1)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn5 = nn.BatchNorm2d(512)
        self.drop5 = nn.Dropout(p=0.5)
        #Original     (when input image size was 64,64)   
        # self.fc_mu = nn.Linear(in_features=512*4, out_features=latent_dims)
        # self.fc_logvar = nn.Linear(in_features=512*4, out_features=latent_dims)
        
        self.fc_mu = nn.Linear(in_features=512*6, out_features=latent_dims) # image after all convolutions is 2,3
        self.fc_logvar = nn.Linear(in_features=512*6, out_features=latent_dims)
            
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
       # x=self.drop1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)))
       # x=self.drop2(x)
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        #x=self.drop3(x)
        
        x = F.leaky_relu(self.bn4(self.conv4(x)))
       # x=self.drop4(x)
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        #x=self.drop5(x)
        
        x = x.view(x.size(0), -1)
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        
        
        self.fc6 = nn.Linear(in_features=latent_dims, out_features=512*6)
            
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 2, 3))
        
        
        self.conv_transpose7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,padding=1,output_padding=(1,0))
        self.bn7 = nn.BatchNorm2d(256)
        
        self.conv_transpose8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        
        
        self.conv_transpose9 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        
        self.conv_transpose10 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn10 = nn.BatchNorm2d(32)
        
        self.conv_transpose11 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        
        
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3,padding=1) 
        
        
        
            
    def forward(self, x):
        x = F.leaky_relu(self.fc6(x))
        
        
        x = self.unflatten(x)
        
       
        x = F.leaky_relu(self.bn7(self.conv_transpose7(x)))
        x = F.leaky_relu(self.bn8(self.conv_transpose8(x)))
        x = F.leaky_relu(self.bn9(self.conv_transpose9(x)))
        x = F.leaky_relu(self.bn10(self.conv_transpose10(x)))
        x = F.leaky_relu(self.bn11(self.conv_transpose11(x)))
        x = F.tanh(self.conv12(x))
        
        
        
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar) #sample the latent vector z
        x_recon = self.decoder(latent_mu) # pass the sampled latent vector z to the decoder. changed to latent_mu for vanilla autoencoder i.e. no sampling
        # return x_recon, latent_mu, latent_logvar #was this previously
        return x_recon,latent,latent_mu,latent_logvar #also return latent sample
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick - compute the latent vector z from the mean and sigma
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
        
#%% Load saved model
model = VariationalAutoencoder()
list_of_files = glob.glob(CONFIG_PATH +'saved_models/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getmtime)
model_file_name = latest_file
model.load_state_dict(torch.load(model_file_name,map_location=('cpu'))) # original mapped location is cpu


#%% Load datasets

train_dataset,test_dataset,target_dataset = load_dataset_dataloader(config)

#%% Evaluate loss on different datasets and extract latent spaces

test_latent_features,test_recon_loss = evaluate_loss_latent(test_dataset, config)
target_latent_features,target_recon_loss = evaluate_loss_latent(target_dataset, config)

plt.figure()
plt.title('reconstruction loss')
plt.plot(test_recon_loss,label='test')
plt.plot(target_recon_loss,label='target ')
plt.legend()
plt.grid()
plt.show(block=False)


# Combine latent spaces of test and target datasets
combined_latent_features = np.concatenate((test_latent_features,target_latent_features),axis=0)

#%% Apply dimensionality redcution on dataset
reduced_latent_features = reduce_dimensions(combined_latent_features,num_components=2)

reduced_latent_features_normalized = MinMaxScaler().fit_transform(reduced_latent_features)
combined_recon = np.concatenate((test_recon_loss,target_recon_loss),axis=0)
combined_recon_normalized = MinMaxScaler().fit_transform(combined_recon.reshape(-1,1))
reduced_latent_features_normalized = np.concatenate((reduced_latent_features_normalized,combined_recon_normalized),axis=1)

# convert reduced features to dataframe

feature_dataframe = pd.DataFrame(reduced_latent_features_normalized)


feature_dataframe["category"] = pd.NaT

feature_dataframe.loc[0:len(test_dataset)-1,['category']] = 'normal'
feature_dataframe.loc[len(test_dataset):,['category']] = 'anomaly'

plt.figure(figsize=(20,20))
g = sns.PairGrid(feature_dataframe,hue='category')
g.map(sns.scatterplot,alpha =0.2)



#%%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')

ax.scatter(reduced_latent_features_normalized[0:len(test_dataset)-1,0],reduced_latent_features_normalized[0:len(test_dataset)-1,1] , reduced_latent_features_normalized[0:len(test_dataset)-1,2])
ax.scatter(reduced_latent_features_normalized[len(test_dataset):,0],reduced_latent_features_normalized[len(test_dataset):,1] , reduced_latent_features_normalized[len(test_dataset):,2],c='red')
ax.view_init(elev=90, azim=270)
