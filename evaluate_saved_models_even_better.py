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

#%%Find best fit distributions for normal (test data) and anomalous (target data)

from fitter import Fitter, get_common_distributions, get_distributions

# f = Fitter(test_recon_loss,
#            distributions=['gamma',
#                           'lognorm',
#                           "beta",
#                           "burr",
#                           "norm"])
f = Fitter(test_recon_loss,distributions=['norm'])
f.fit()
f.summary()
f.get_best(method = 'sumsquare_error')


#%% Combine latent spaces of test and target datasets
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
#%%
# plt.figure(figsize=(20,20))
# g = sns.PairGrid(feature_dataframe,hue='category')
# g.map(sns.scatterplot,alpha =0.2)



#%% plot latent space in 3D
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')

ax.scatter(reduced_latent_features_normalized[0:len(test_dataset)-1,0],reduced_latent_features_normalized[0:len(test_dataset)-1,1] , reduced_latent_features_normalized[0:len(test_dataset)-1,2])
ax.scatter(reduced_latent_features_normalized[len(test_dataset):,0],reduced_latent_features_normalized[len(test_dataset):,1] , reduced_latent_features_normalized[len(test_dataset):,2],c='red')
ax.view_init(elev=90, azim=270)

#%% Implement Local Outlier Factor (LOF)

# from sklearn.neighbors import LocalOutlierFactor

# X_inliers = reduced_latent_features_normalized[0:len(test_dataset)-1,:]
# X_outliers = reduced_latent_features_normalized[len(test_dataset):,:]
# X = reduced_latent_features_normalized
# n_outliers = len(X_outliers)
# ground_truth = np.ones(len(reduced_latent_features_normalized), dtype=int)
# ground_truth[-n_outliers:] = -1


# # fit the model for outlier detection (default)

# clf = LocalOutlierFactor(n_neighbors=4,contamination=0.1,metric="manhattan")
# # use fit_predict to compute the predicted labels of the training samples
# # (when LOF is used for outlier detection, the estimator has no predict,
# # decision_function and score_samples methods).
# y_pred = clf.fit_predict(MinMaxScaler().fit_transform(reduced_latent_features_normalized))
# n_errors = (y_pred != ground_truth).sum()
# X_scores = clf.negative_outlier_factor_





# plt.figure(figsize=(20,20))
# plt.title("Local Outlier Factor (LOF)")
# plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
# plt.scatter(X_outliers[:, 0], X_outliers[:, 1],marker="x", s=20,color="m", label="true anomalies")
# # plot circles with radius proportional to the outlier scores
# radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
# plt.scatter(
#     X[:, 0],
#     X[:, 1],
#     s=400 * radius,
#     edgecolors="r",
#     facecolors="none",
#     label="Outlier scores",alpha=0.3,
# )





# plt.axis("tight")
# # plt.xlim((-5, 5))
# # plt.ylim((-5, 5))
# plt.xlabel("prediction errors: %d" % (n_errors))
# legend = plt.legend(loc="upper left")
# legend.legendHandles[0]._sizes = [10]
# legend.legendHandles[1]._sizes = [20]
# plt.show()


# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# precision = precision_score(ground_truth, y_pred)
# recall = recall_score(ground_truth,y_pred)

# print('precision score: ',precision)
# print('recall score: ',recall)


#%% Use outlier detection methods
from sklearn.ensemble import IsolationForest



X_inliers = reduced_latent_features_normalized[0:len(test_dataset)-1,:]
X_outliers = reduced_latent_features_normalized[len(test_dataset):,:]
X = reduced_latent_features_normalized
n_outliers = len(X_outliers)
ground_truth = np.ones(len(reduced_latent_features_normalized), dtype=int)
ground_truth[-n_outliers:] = -1


# fit the model for outlier detection (default)

clf = IsolationForest(n_estimators=200)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(MinMaxScaler().fit_transform(X))
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.decision_function(X)





plt.figure(figsize=(20,20))
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
plt.scatter(X_outliers[:, 0], X_outliers[:, 1],marker="x", s=20,color="m", label="true anomalies")
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=400 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",alpha=0.3,
)





plt.axis("tight")
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc="upper left")
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score,confusion_matrix

ground_truth_copy = np.copy(ground_truth)
y_pred_copy = np.copy(y_pred)

ground_truth_copy[ground_truth_copy==1]=0
ground_truth_copy[ground_truth_copy==-1]=1

y_pred_copy[y_pred_copy==1]=0
y_pred_copy[y_pred_copy==-1]=1

conf_matrix = confusion_matrix(ground_truth_copy,y_pred_copy)
tn, fp, fn, tp = conf_matrix.ravel()

precision = precision_score(ground_truth_copy, y_pred_copy,average=None)
recall = recall_score(ground_truth_copy,y_pred_copy,average=None)

print('true positives: ',tp)
print('false positives: ',fp)
print('true negatives: ',tn)
print('false negatives: ',fn)

print('precision score: ',precision)
print('recall score: ',recall)
print(conf_matrix)
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
#pca_combined = np.concatenate((pca_components,target_pca_components))
db = DBSCAN(eps=0.03, min_samples=6).fit(reduced_latent_features_normalized[:,:2])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

cluster_anomaly_predictions = np.zeros(len(labels))
cluster_anomaly_predictions[labels==-1] = 1

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# # =============================================================================
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print(
#      "Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels)
#  )
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
# # =============================================================================


plt.figure()
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = reduced_latent_features_normalized[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = reduced_latent_features_normalized[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
plt.scatter(reduced_latent_features_normalized[len(test_dataset):,0],reduced_latent_features_normalized[len(test_dataset):,1] ,s=165,c='red')
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


cluster_conf_matrix = confusion_matrix(ground_truth_copy,cluster_anomaly_predictions)
tn_cluster, fp_cluster, fn_cluster, tp_cluster = cluster_conf_matrix.ravel()

precision_cluster = precision_score(ground_truth_copy,cluster_anomaly_predictions,average=None)
recall_cluster = recall_score(ground_truth_copy,cluster_anomaly_predictions,average=None)

print('cluster true positives: ',tp_cluster)
print('cluster false positives: ',fp_cluster)
print('cluster true negatives: ',tn_cluster)
print('cluster false negatives: ',fn_cluster)

print('cluster precision score: ',precision_cluster)
print('cluster recall score: ',recall_cluster)
print(cluster_conf_matrix)

#%% obtain optimal epsilon value
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(reduced_latent_features_normalized[:,:2])
distances, indices = neighbors_fit.kneighbors(reduced_latent_features_normalized[:,:2])

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

#%% select threshold for reconstruction error

fitter = Fitter(combined_recon_normalized,distributions=['norm'])
fitter.fit()
fitter.summary()
recon_stats = fitter.get_best(method = 'sumsquare_error')

recon_avg = recon_stats['norm']['loc']
recon_std = recon_stats['norm']['scale']

num_std_dev = 0.6
threshold = recon_avg + num_std_dev*recon_std

recon_anomaly_mask = (combined_recon_normalized>=threshold)
cluster_anomaly_mask = cluster_anomaly_predictions.astype(dtype=(bool))

joint_prediction_mask = (cluster_anomaly_mask.reshape(-1,1) & recon_anomaly_mask)

num_remaining = sum(joint_prediction_mask)

print('number of remaining points: ',num_remaining)

remaining_data = reduced_latent_features_normalized[:,:2][joint_prediction_mask]

maskArr1 = np.ma.masked_array(reduced_latent_features_normalized[:,0], mask =joint_prediction_mask)
maskArr2 = np.ma.masked_array(reduced_latent_features_normalized[:,1], mask =joint_prediction_mask)
joint_predictions = np.ma.concatenate([np.ma.expand_dims(maskArr1,axis=1),np.ma.expand_dims(maskArr2,axis=1)],axis=1)
#%%
plt.figure(figsize=(15,15))
plt.title("reconstruction error scores")
plt.scatter(X[:, 0], X[:, 1], c=X[:,2],s=10.0, label="Data points")
plt.colorbar()
#plt.scatter(X_outliers[:, 0], X_outliers[:, 1],c=X_outliers[:,2], s=10, label="true anomalies")

#%%
plt.figure(figsize=(15,15))
plt.title("reconstruction error scores")
plt.scatter(joint_predictions[:, 0], joint_predictions[:, 1],s=10.0, label="Data points")
plt.colorbar()
