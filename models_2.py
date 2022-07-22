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
# latent_dims = Main.config['model_params']['latent_dim'] # was hardcoded 256 previously
latent_dims = 256
capacity = 64

# variational_beta = Main.config['model_params']['kld_weight']

# variational_beta = 1.0

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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn1 = nn.BatchNorm2d(32)
        
        
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=2,padding=1) # input_dim = [3,224,280];out: [96,54,68]
        self.bn5 = nn.BatchNorm2d(512)
        
        #Original     (when input image size was 64,64)   
        # self.fc_mu = nn.Linear(in_features=512*4, out_features=latent_dims)
        # self.fc_logvar = nn.Linear(in_features=512*4, out_features=latent_dims)
        
        self.fc_mu = nn.Linear(in_features=512*6, out_features=latent_dims) # image after all convolutions is 2,3
        self.fc_logvar = nn.Linear(in_features=512*6, out_features=latent_dims)
            
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
       
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
        x_recon = self.decoder(latent) # pass the sampled latent vector z to the decoder
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
    recon_loss = torch.nn.MSELoss(reduction='sum') (recon_x,x)
    
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