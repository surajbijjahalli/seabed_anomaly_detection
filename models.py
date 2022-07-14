#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:42:51 2022

@author: surajb
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

'''Parameter Settings
-------------------
'''

# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)

# latent dims might become a problem in terms of blurriness of images. This prob;em
#was encountered here: https://stackoverflow.com/questions/63976757/vae-reconstructed-images-are-extremely-blurry
# solutions may include increasing the number of latent dimensions or in gradually converging to the bottleneck. In that case, perhaps something to look at
# would be something like t-SNE to visualize high-dimensional latent space. The effect of latent dimensions are reiterated by Ava Soleimany in the MIT deep learning youtube lecture (https://www.youtube.com/watch?v=rZufA635dq4)
latent_dims = 256

capacity = 64

variational_beta = 1

''' Some ideas for the number of layers and hyperparams may be taken from https://github.com/AntixK/PyTorch-VAE

'''

'''
VAE Definition
-----------------------

We use a convolutional encoder and decoder, which generally gives better performance than fully connected versions that have the same number of parameters.

In convolution layers, we increase the channels as we approach the bottleneck, but note that the total number of features still decreases, since the channels increase by a factor of 2 in each convolution, but the spatial size decreases by a factor of 4.

Kernel size 4 is used to avoid biasing problems described here: https://distill.pub/2016/deconv-checkerboard/
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4) # input_dim = [3,224,280];out: [96,54,68]
        self.pool1 = nn.MaxPool2d(3, stride=(2)) # input_dim = [96,54,68];out: [96,26,33]
        self.drop1 = nn.Dropout(p=0.05)
        self.bn1 = nn.BatchNorm2d(96)
        
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size= 5,padding=2) # input_dim = [96,26,33];out: [256,26,33]
        self.pool2 = nn.MaxPool2d(3, stride=2) # input_dim = [256,26,33];out: [256,12,16]
        self.drop2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size= 3,padding=1) # input_dim = [256,12,16];out: [384,12,16]
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size= 3,padding=1) # input_dim = [384,12,16];out: [384,12,16]
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size= 3,padding=1)# input_dim = [384,12,16];out: [256,12,16]
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool5 = nn.MaxPool2d(3,stride= 2) # input_dim = [256,12,16];out: [256,5,7]
        self.drop5 = nn.Dropout(p=0.1)
        
        self.fc6 = nn.Linear(in_features=256*5*7,out_features=4480)
        self.drop6 = nn.Dropout(p=0.1)
        
        
        
        self.fc7 = nn.Linear(in_features=4480,out_features=2240)
        self.drop7 = nn.Dropout(p=0.1)
        
        self.fc8 = nn.Linear(in_features=2240,out_features=1120)
        self.drop8 = nn.Dropout(p=0.1)
        
        self.fc_mu = nn.Linear(in_features=1120, out_features=latent_dims)
       # self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=1120, out_features=latent_dims)
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
       # x = self.drop2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = self.pool5(x)
        #x = self.drop5(x)
        
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        #x = self.drop6(x)
        
        x = F.relu(self.fc7(x))
        #x = self.drop7(x)
        
        x = F.relu(self.fc8(x))
        #x = self.drop8(x)
        
    
        
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc9 = nn.Linear(in_features=latent_dims, out_features=1120)
        self.drop9 = nn.Dropout(p=0.1)
        
        self.fc10 = nn.Linear(in_features=1120, out_features=2240)
        self.drop10 = nn.Dropout(p=0.1)
        
        self.fc11 = nn.Linear(in_features=2240, out_features=4480)
        self.drop11 = nn.Dropout(p=0.1)
        
        self.fc12 = nn.Linear(in_features=4480, out_features=8960)
        self.drop12 = nn.Dropout(p=0.1)
        
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 5, 7))
        
        self.upsample13 = nn.Upsample(size=(12,16))
        self.conv_transpose13 = nn.ConvTranspose2d(in_channels=256, out_channels=384, kernel_size=3, stride=1,padding=1)
        self.drop13 = nn.Dropout(p=0.1)
        
        
        self.conv_transpose14 = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=3, stride=1,padding=1)
        self.drop14 = nn.Dropout(p=0.1)
        
        self.conv_transpose15 = nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=3, stride=1,padding=1)
        self.drop15 = nn.Dropout(p=0.1)
        
        self.upsample16 = nn.Upsample(size=(26,33))
        self.conv_transpose16 = nn.ConvTranspose2d(in_channels=256, out_channels=96, kernel_size=5, padding=2)
        self.drop16 = nn.Dropout(p=0.05)
        
        
        self.upsample17 = nn.Upsample(size=(54,68))
        self.conv_transpose17 = nn.ConvTranspose2d(in_channels=96, out_channels=3, kernel_size=12, stride=4)
        
        #self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = F.relu(self.fc9(x))
        #x = self.drop9(x)
        
        x = F.relu(self.fc10(x))
        #x = self.drop10(x)
        
        x = F.relu(self.fc11(x))
        #x = self.drop11(x)
                
        x = F.relu(self.fc12(x))
        #x = self.drop12(x)
        
        x = self.unflatten(x)
        
        x = self.upsample13(x)
        x = F.relu(self.conv_transpose13(x))
        #x = self.drop13(x)
        
        x = F.relu(self.conv_transpose14(x))
        #x = self.drop14(x)
        
        x = F.relu(self.conv_transpose15(x))
        #x = self.drop15(x)
        
        
        x = self.upsample16(x)
        x = F.relu(self.conv_transpose16(x))
        #x = self.drop16(x)
        
        x = self.upsample17(x)
        x = F.relu(self.conv_transpose17(x))
        
       # x = self.conv_transpose1(x)
       # x = x.view(x.size(0), capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
       # x = F.relu(self.conv2(x))
       # x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar) #sample the latent vector z
        x_recon = self.decoder(latent) # pass the sampled latent vector z to the decoder
        return x_recon, latent_mu, latent_logvar
    
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
        
        
def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    # Note to self - a different reconstruction loss term may have to be used here e.g. a squared-Euclidean distance F.mse_loss
    
    # another way of specifying the loss is mentioned here: https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
    #criterion = nn.BCELoss(reduction='sum')
    
    # A good discussion on the use of BCE vs MSE is here: https://github.com/pytorch/examples/issues/399 . Essentially BCE was used in
    # original VAE paper because the decoder hada sigmoid at the end which can be viewed as a 'Bernoulli distribution'
    
    #recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    
    # Mean squared error loss - by default, the loss is averaged over number of elements,
    # alternatives for reduction are 'none' or 'sum'
    recon_loss = torch.nn.MSELoss() (recon_x,x)
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image. 
    
    # The signs appear to be reversed. Also the .exp() for the logvar 
    # This site appears to confirm summing divergence over latent dimensions i.e. dim = 1 (https://kvfrans.com/variational-autoencoders-explained/)
    # This site confirms that the divergence is summed over latent dimensions https://leenashekhar.github.io/2019-01-30-KL-Divergence/
    # further confirmation that the divergence is summed over latent dimensions (https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes)
    # A possible option is to use the built-in function (https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
    kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0) # mostly consistent with https://arxiv.org/pdf/1907.08956v1.pdf
    
    #kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    
    
    
    return recon_loss + variational_beta * kldivergence, recon_loss, kldivergence