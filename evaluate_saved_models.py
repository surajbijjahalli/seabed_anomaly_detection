#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:52:35 2022

@author: surajb
"""
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
        
        self.fc_mu = nn.Linear(in_features=512*6, out_features=64) # image after all convolutions is 2,3
        self.fc_logvar = nn.Linear(in_features=512*6, out_features=64)
            
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
        
        # Original     (when input image size was 64,64)
        # self.fc6 = nn.Linear(in_features=latent_dims, out_features=512*4)
            
        
        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 2, 2))
        
        self.fc6 = nn.Linear(in_features=64, out_features=512*6)
            
        
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
        
# A very good resource on VAE theory is here - could be useful in making sure input-output dimensions
# are consistent (https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf)            
        
        
def vae_loss(recon_x, x, mu, logvar,variational_beta):
    
    
    # Mean squared error loss - by default, the loss is averaged over number of elements,
    # alternatives for reduction are 'none' or 'sum'
    recon_loss = torch.nn.MSELoss(reduction='sum') (recon_x,x)
    #recon_loss  = 0.4*torch.nn.L1Loss(reduction='mean') (recon_x,x)+(1-ssim(recon_x,x))
    
    kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0) # mostly consistent with https://arxiv.org/pdf/1907.08956v1.pdf
    
        
    
    return recon_loss + variational_beta * kldivergence, recon_loss, kldivergence




class Evaluate_model():
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self,experiment_name,normal_dataset,anomaly_dataset):
        
        self.experiment_name = str(experiment_name)
        CONFIG_PATH = str(experiment_name) + '/'
        
        self.normal_dataset = normal_dataset
        self.anomaly_dataset = anomaly_dataset
       # self.config = yaml.safe_load(open(os.path.join(CONFIG_PATH, self.experiment_name + '.yaml')))
        self.config = yaml.safe_load(open( self.experiment_name + '.yaml'))
        
        
        self.latent_dims = self.config['model_params']['latent_dim']
        self.variational_beta = self.config['model_params']['kld_weight']
        
        
        self.model = VariationalAutoencoder()

    
        
        # Change vae params according to config file
        self.model.encoder.fc_mu.out_features = self.latent_dims
        self.model.encoder.fc_logvar.out_features = self.latent_dims
        self.model.decoder.fc6.in_features = self.latent_dims

        #best_model.load_state_dict(torch.load(saved_models_path+'/'+ best_model_name,map_location=('cpu'))) # original mapped location is cpu

        #self.model.load_state_dict(torch.load(saved_models_path+'/'+ best_model_name,map_location=device)) # new location is device

        
        
    def print_model(self):
        print(self.model)



# =============================================================================
# def __call__(self, sample):
#         image, key_pts = sample['image'], sample['keypoints']
# 
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
# 
#         new_h, new_w = int(new_h), int(new_w)
# 
#         img = cv2.resize(image, (new_w, new_h))
#         
#         # scale the pts, too
#         key_pts = key_pts * [new_w / w, new_h / h]
# 
#         return {'image': img, 'keypoints': key_pts}
# =============================================================================

eval_model = Evaluate_model(10, 'normal_dataset', 'anomaly_dataset')