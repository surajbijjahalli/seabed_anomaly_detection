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

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


#config = load_config("baseline.yaml")
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
        
# A very good resource on VAE theory is here - could be useful in making sure input-output dimensions
# are consistent (https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L17_vae__slides.pdf)            
        
        
def vae_loss(recon_x, x, mu, logvar,variational_beta):
    
    
    # Mean squared error loss - by default, the loss is averaged over number of elements,
    # alternatives for reduction are 'none' or 'sum'
    recon_loss = torch.nn.MSELoss(reduction='sum') (recon_x,x)
    #recon_loss  = 0.4*torch.nn.L1Loss(reduction='mean') (recon_x,x)+(1-ssim(recon_x,x))
    
    kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0) # mostly consistent with https://arxiv.org/pdf/1907.08956v1.pdf
    
        
    
    return recon_loss + variational_beta * kldivergence, recon_loss, kldivergence


model = VariationalAutoencoder()
list_of_files = glob.glob(CONFIG_PATH +'saved_models/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getmtime)
model_file_name = latest_file

 
model.load_state_dict(torch.load(model_file_name,map_location=('cpu'))) # original mapped location is cpu

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


from data_load import create_datasets,RescaleCustom, RandomCropCustom, NormalizeNew, ToTensorCustom, RandomRotateCustom, RandomHorizontalFlip,RandomVerticalFlip,NewColorJitter
from torchvision import transforms, utils
from data_load import MarineBenthicDataset

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



#initialize arrays for storing loss on training and test datasets
train_test_loss_over_time = np.zeros(len(train_dataset))
train_test_recon_loss_over_time = np.zeros(len(train_dataset))
train_test_kl_loss_over_time = np.zeros(len(train_dataset))



test_loss_over_time = np.zeros(len(test_dataset))
test_recon_loss_over_time = np.zeros(len(test_dataset))
test_kl_loss_over_time = np.zeros(len(test_dataset))

#initialize arrays for storing loss on target dataset
target_loss_over_time = np.zeros(len(target_dataset))
target_recon_loss_over_time = np.zeros(len(target_dataset))
target_kl_loss_over_time = np.zeros(len(target_dataset))



latent_space_training_set = np.zeros((len(train_dataset),int(config['model_params']['latent_dim'])))

latent_space_target_set = np.zeros((len(target_dataset),int(config['model_params']['latent_dim'])))

latent_space_test_set = np.zeros((len(test_dataset),int(config['model_params']['latent_dim'])))



model.eval()


for idx, data in enumerate(test_dataset):
    
    # get the input images in each batch and their corresponding names
    images = data['image']
    name = data['name']
    
    #images = images.to(device)
    images = torch.unsqueeze(images,0)
    # forward pass the images through the network
    
    image_recon,latent_sample_z,mu_z,log_var_z = model(images)
    latent_space_test_set[idx,:] = log_var_z.detach().cpu().numpy()


for idx, data in enumerate(target_dataset):
    
    # get the input images in each batch and their corresponding names
    images = data['image']
    name = data['name']
    
    #images = images.to(device)
    images = torch.unsqueeze(images,0)
    # forward pass the images through the network
    
    image_recon,latent_sample_z,mu_z,log_var_z = model(images)
    latent_space_target_set[idx,:] = log_var_z.detach().cpu().numpy()

from sklearn.preprocessing import StandardScaler,MinMaxScaler

# change the latent space array to either the train dataset (latent_space_array) or the test dataset(latent_space_test_set_array) as per requirement
latent_space_test_set = StandardScaler().fit_transform(latent_space_test_set) # was fit _transform(latent_space_array) before - when trying to fit the training set
latent_space_target_set = StandardScaler().fit_transform(latent_space_target_set)
from sklearn.decomposition import PCA


pca = PCA(n_components=3,whiten=False)
pca_components = pca.fit_transform(latent_space_test_set)

target_pca_components = pca.transform(latent_space_target_set)

plt.figure()
plt.scatter(pca_components[:,0],pca_components[:,1],alpha=0.2,c='green')
plt.scatter(target_pca_components[:,0],target_pca_components[:,1],alpha=0.2,c='red')
plt.grid()

plt.figure()
plt.scatter(pca_components[:,1],pca_components[:,2],alpha=0.2,c='green')
plt.scatter(target_pca_components[:,1],target_pca_components[:,2],alpha=0.2,c='red')
plt.grid()

plt.figure()
plt.scatter(pca_components[:,0],pca_components[:,2],alpha=0.2,c='green')
plt.scatter(target_pca_components[:,0],target_pca_components[:,2],alpha=0.2,c='red')
plt.grid()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_components[:,0],pca_components[:,1],pca_components[:,2], alpha=0.2,c='green')
ax.scatter(target_pca_components[:,0],target_pca_components[:,1],target_pca_components[:,2], alpha=0.2,c='red')
#%%
import seaborn as sns

from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import metrics
# =============================================================================
# plt.figure()
# somefig, someax = plt.subplots(figsize=(6, 6))
# sns.scatterplot(
#     data=pca_components,
#     x=pca_components[:,0],
#     y=pca_components[:,1],
#     color="g",
#     ax=someax,alpha=0.4
# )
# 
# sns.scatterplot(
#     data=pca_components,
#     x=target_pca_components[:,0],
#     y=target_pca_components[:,1],
#     color="r",
#     ax=someax,alpha=0.4
# )
# sns.kdeplot(
#     data=pca_components,
#     x=pca_components[:,0],
#     y=pca_components[:,1],
#     levels=50,
#     alpha=0.4,
#     ax=someax,
# )
# =============================================================================

# %%
pca_combined = np.concatenate((pca_components,target_pca_components))
db = DBSCAN(eps=0.6, min_samples=4).fit(pca_combined)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# =============================================================================
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print(
#     "Adjusted Mutual Information: %0.3f"
#     % metrics.adjusted_mutual_info_score(labels_true, labels)
# )
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
# =============================================================================

plt.figure()
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = pca_combined[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = pca_combined[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()