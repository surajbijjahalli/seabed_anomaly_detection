#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:31:48 2022

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



from torch.utils.tensorboard import SummaryWriter

# Logging metadata
import neptune.new as neptune
from neptune.new.types import File

import yaml

## specify command line arguments to be supplied by user






# Create the parser
my_parser = argparse.ArgumentParser(description='train and test a model using a specified config file')

# Add the arguments
my_parser.add_argument('Filename',
                       metavar='filename',
                       type=str,
                       help='the name of the config file which specifies hyperparams')

# Execute the parse_args() method
args = my_parser.parse_args()

config_file_name = args.Filename



# Create neptune run object for logging metrics and metadata
# NEPTUNE_API_TOKEN = "<api-token-here>"
run = neptune.init(
    project="acfr-marine/seabed-anomaly-detection",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGRhZjQzOS1mMTE2LTQ3NzUtYWEwYS1hNDg0ZDAxOTVhZTgifQ==",
)  # your credentials

#run = neptune.init(project='surajbijjahalli/marine-anomaly-detection',api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMGRhZjQzOS1mMTE2LTQ3NzUtYWEwYS1hNDg0ZDAxOTVhZTgifQ==')

# folder to load config file
CONFIG_PATH = "config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


#config = load_config("baseline.yaml")
config = load_config(config_file_name)

# export params from config file to Neptune    
run['parameters'] = config 

model_name = config['model_params']['model_name']

output_path = config['experiment_params']['output_path']
experiment_name = config['experiment_params']['experiment_name']
experiment_path = os.path.join(output_path, experiment_name) 
training_log_path = os.path.join(experiment_path,'training_log')
saved_models_path = os.path.join(experiment_path,'saved_models')
test_results_path = os.path.join(experiment_path,'test_results')

if not os.path.exists(training_log_path):
    os.makedirs(training_log_path)
    print("folder '{}' created ".format(training_log_path))
else:
    print("folder {} already exists".format(training_log_path))
    
    
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)
    print("folder '{}' created ".format(saved_models_path))
else:
    print("folder {} already exists".format(saved_models_path))
    
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)
    print("folder '{}' created ".format(test_results_path))
else:
    print("folder {} already exists".format(test_results_path))
    


# os.mkdir(experiment_path) 
# print("Directory '% s' created" % experiment_path) 

writer = SummaryWriter()
#%% Grab paths to training images

#train_data_path = '/media/surajb/Extreme SSD/marine_dataset_Jervis_2021/r20210407_061916_NG106_hyams_depth_gated_target/raw_targetless_imgs/alp/converted_images'



# train_data_path = 'dataset_split/train/data'
# test_data_path = 'dataset_split/test/data'
# val_data_path = 'dataset_split/val/data'


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


#%% Import custom classes and define transform


# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import MarineBenthicDataset, visualize_output_loader,visualize_raw_data_loader, encoder_sample_output,visualize_vae_output,visualize_vae_output_eval
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import create_datasets,RescaleCustom, RandomCropCustom, NormalizeNew, ToTensorCustom, RandomRotateCustom, RandomHorizontalFlip,RandomVerticalFlip,ColorJitter


# Define data transforms for training, validation and testing 
# train_transform = transforms.Compose([RandomCropCustom(64),ToTensorCustom(),RandomHorizontalFlip(),RandomVerticalFlip()])
# valid_transform = transforms.Compose([RandomCropCustom(64),ToTensorCustom()])

train_transform = transforms.Compose([RescaleCustom(64), ToTensorCustom(),RandomHorizontalFlip(),RandomVerticalFlip()])
valid_transform = transforms.Compose([RescaleCustom(64),ToTensorCustom()])

test_transform  = transforms.Compose([RescaleCustom(64),ToTensorCustom()])
target_transform  = transforms.Compose([RescaleCustom(64),ToTensorCustom()])





#%% Create datasets and dataloaders
batch_size = config['data_params']['train_batch_size']
JB_train_loader,JB_test_loader,JB_val_loader,JB_train_dataset,JB_test_dataset,JB_valid_dataset,raw_dataset,raw_loader,target_dataset = create_datasets(batch_size=batch_size, train_image_paths = train_image_paths,test_image_paths=test_image_paths,val_image_paths=val_image_paths,target_image_paths=target_image_paths,
                    train_transform=train_transform,test_transform=test_transform,valid_transform=valid_transform,target_transform=target_transform)

# print some stats about the dataset
print('Length of training dataset: ', len(JB_train_dataset))
print('Length of validation dataset: ', len(JB_valid_dataset))
print('Length of test dataset: ', len(JB_test_dataset))
print('Length of target dataset: ', len(target_dataset))

#print('Original Image shape: ', JervisDataset[0]['image'].shape)
print('Original image shape', raw_dataset[0]['image'].shape)
print('Transformed Image shape: ', JB_train_dataset[0]['image'].shape)

#plot sample images from dataloader - This is to observe that transforms are 
#correctly applied before training
      
visualize_raw_data_loader(raw_loader)
visualize_output_loader(JB_train_loader)


#%% Import VAE model

from models_2 import VariationalAutoencoder,vae_loss



vae = VariationalAutoencoder()

# Change vae params according to config file
vae.encoder.fc_mu.out_features = config['model_params']['latent_dim']
vae.encoder.fc_logvar.out_features = config['model_params']['latent_dim']
vae.decoder.fc6.in_features = config['model_params']['latent_dim']

variational_beta = config['model_params']['kld_weight']



# %% Initialize weights

# Define function for initializing network weights
#Xavier initialization - All weights of a layer are picked from a zero-mean normal distribution with variance 
#being a function of the number of inputs to that layer (i.e. the number of nodes in the previous layer)

def init_weights(layer):
    torch.nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
    layer.bias.data.fill_(0.01) # To initialize bias parameters


# Call initialization function on encoder weights

init_weights(vae.encoder.conv1)
init_weights(vae.encoder.conv2)
init_weights(vae.encoder.conv3)
init_weights(vae.encoder.conv4)
init_weights(vae.encoder.conv5)

init_weights(vae.encoder.fc_mu)
init_weights(vae.encoder.fc_logvar)

# Initialize decoder weights

init_weights(vae.decoder.fc6)

init_weights(vae.decoder.conv_transpose7)
init_weights(vae.decoder.conv_transpose8)
init_weights(vae.decoder.conv_transpose9)
init_weights(vae.decoder.conv_transpose10)
init_weights(vae.decoder.conv_transpose11)
init_weights(vae.decoder.conv12)

# Get number of trainable parameters in the model
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

print(vae)



# Pass a batch of images from the train loader through the model before training
       
orig_image_sample,orig_image_sample_name,recon_image_sample,latent_sample_vector,latent_mu_sample,latent_logvar_sample = encoder_sample_output(vae,JB_train_loader)

# Visualize output from VAE

visualize_vae_output(orig_image_sample, recon_image_sample)   


#%% Model training

# Set up functions for regulating the training routine

#import loss function - sum of reconstruction error and kl divergence

from training_utils import EarlyStopping,write_list_to_file


# Define params for training
use_gpu = config['trainer_params']['use_gpu']
num_epochs = config['trainer_params']['max_epochs']
learning_rate = config['trainer_params']['LR']
weight_decay = config['trainer_params']['weight_decay']
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define scheduler to reduce learning rate when a validation loss has stopped improving - default reduction is 0.1
plateau_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  patience=1, verbose=True)

# The input arguments of the scheduler are:
#optimizer - the defined optimizer for adjusting weights
#Mode: 'min'/'max' - whether to reduce learning rate based on whether a metric is decreasing('min') or increasing('max')
#patience: Number of epochs with no improvement (in metric) after which learning rate will be reduced 


num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
   

# samp_images = next(iter(JB_train_loader))['image']
# writer.add_graph(vae, samp_images)

# Define validation loss
def validation_loss(valid_loader, vae):  # pass in validation loader and the model as input arguments
    # set the module to evaluation mode
    vae.eval()
    loss = 0.0
    running_loss = 0.0
    # iterate through the test dataset
    for i, batch in enumerate(valid_loader): #iterate through validation loader batch by batch
        # get sample data: images and ground truth keypoints
        images = batch['image']
        name = batch['name']
        
        
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        # forward pass to get net output
        images_recon,latent_z, latent_mu, latent_logvar = vae(images)
        
        # calculate the loss 
        loss,recon_val_loss,kl_val_loss = vae_loss(images_recon, images, latent_mu, latent_logvar,variational_beta)
        running_loss += loss.item()
    avg_loss = running_loss/(i+1)
    vae.train()
    return avg_loss





  
    
# Train model

train_loss_over_time = [] # to track the loss as the network trains
val_loss_over_time = [] # to track the validation loss as the network trains
val_loss_min = np.Inf
early_stopping = EarlyStopping()
learning_rate_list = [] # to track learning rate when a scheduler is used
vae.train()

print('Running experiment: %s ' % experiment_name +'on model %s' % model_name) 
print('Number of trainable parameters: %d' % num_params) 



for epoch in range(num_epochs):
    running_train_loss = 0.0
    avg_val_loss = 0.0
    avg_train_loss = 0.0
    
    running_train_recon_loss = 0.0
    running_train_kl_loss = 0.0
    
    avg_train_recon_loss = 0.0
    avg_train_kl_loss = 0.0

       # train on batches of data, assumes you already have train_loader
    for batch_i, data in enumerate(JB_train_loader):
        
        # get the input images in each batch and their corresponding names
        images = data['image']
        name = data['name']
        
        images = images.to(device)
        
        # forward pass the images through the network
        
        image_recon,latent_sample_z,mu_z,log_var_z = vae(images)
        
        # Calculate loss
        loss,recon_train_loss,kl_train_loss = vae_loss(image_recon, images, mu_z, log_var_z,variational_beta)
        
       # zero the parameter (weight) gradients
        optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
        loss.backward()

            # update the weights
        optimizer.step()
        
            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
        running_train_loss += loss.item()
        running_train_recon_loss += recon_train_loss.item()
        running_train_kl_loss += kl_train_loss.item() 
        
    # validate the model using the validation dataset
        
    avg_val_loss = validation_loss(JB_val_loader, vae) # returns loss averaged over all mini-batches in the validation loader
        
    
    
    # Average the training loss over all the mini-batches in the training loader for one epoch
        
    avg_train_loss = running_train_loss/len(JB_train_loader)
    avg_train_recon_loss = running_train_recon_loss/len(JB_train_loader)
    avg_train_kl_loss = running_train_kl_loss/len(JB_train_loader)
    
    
        
    train_loss_over_time.append(avg_train_loss)
    val_loss_over_time.append(avg_val_loss)
    
    fig_export = visualize_vae_output(images, image_recon)
    learning_rate_over_time = optimizer.state_dict()['param_groups'][0]['lr']
    learning_rate_list.append(learning_rate_over_time)
    
    #export loss metrics and image sample reconstructions to Neptune
    run['metrics/train_loss'].log(avg_train_loss)
    run['metrics/val_loss'].log(avg_val_loss)
    run['metrics/recon_loss'].log(avg_train_recon_loss)
    run['metrics/kl_loss'].log(avg_train_kl_loss)
    
    
    run["predictions/recon_imgs"].upload(File.as_image(fig_export))
    
    print('Epoch:', epoch + 1,'/',num_epochs, 'Avg. Training Loss:',avg_train_loss, 'Avg. Validation Loss:',avg_val_loss)
    print('Epoch:', epoch + 1,'/',num_epochs, 'Avg. reconstruction training Loss:',avg_train_recon_loss, 'Avg. reconstruction kl Loss:',avg_train_kl_loss)
    
    
    
    # save model if validation loss has decreased
    if avg_val_loss <= val_loss_min:
       checkpoint_name = str(epoch+1) 
       print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min,avg_val_loss))
       torch.save(vae.state_dict(), saved_models_path +'/' +'checkpoint'+checkpoint_name+'.pt')
       best_model_name = 'checkpoint'+checkpoint_name+'.pt'
       val_loss_min = avg_val_loss
    
    

# write loss curves to file
write_list_to_file(train_loss_over_time, training_log_path+ '/'+ 'Training_Loss.csv')
write_list_to_file(val_loss_over_time, training_log_path+ '/'+ 'Validation_Loss.csv')   
                        

print('Finished training')


   




#%% Evaluate trained model

plt.figure()


plt.plot(val_loss_over_time,label='Validation loss')
plt.plot(train_loss_over_time,label='Training loss')
plt.legend()
plt.grid()

print('model with lowest validation loss: checkpoint',best_model_name)


print('cleaned up workspace')

#%%
best_model = VariationalAutoencoder()
# Change vae params according to config file
best_model.encoder.fc_mu.out_features = config['model_params']['latent_dim']
best_model.encoder.fc_logvar.out_features = config['model_params']['latent_dim']
best_model.decoder.fc6.in_features = config['model_params']['latent_dim']

#best_model.load_state_dict(torch.load(saved_models_path+'/'+ best_model_name,map_location=('cpu'))) # original mapped location is cpu

best_model.load_state_dict(torch.load(saved_models_path+'/'+ best_model_name,map_location=device)) # new location is device

print(best_model)

'''
#%% Evaluate loss of trained model on test and train datasets
best_model.eval()

#initialize arrays for storing loss on training and test datasets
train_test_loss_over_time = []
train_test_recon_loss_over_time = []
train_test_kl_loss_over_time = []



test_loss_over_time = []
test_recon_loss_over_time = []
test_kl_loss_over_time = []

latent_space_training_set = []

for test_sample in JB_test_dataset:
    test_sample_name = test_sample['name']
    
    test_sample_image = test_sample['image']
    
    
    test_sample_image = torch.unsqueeze(test_sample_image,0)
    
    # # # forward pass the images through the network

    test_sample_image_recon,test_sample_z_vector,test_sample_mu_z,test_sample_log_var_z = best_model(test_sample_image)
            
    # Calculate loss
    test_sample_total_loss,test_sample_recon_loss,test_sample_kl_loss = vae_loss(test_sample_image_recon, test_sample_image, test_sample_mu_z, test_sample_log_var_z,variational_beta)
    
    
    test_loss_over_time.append(test_sample_total_loss.item())
    test_recon_loss_over_time.append(test_sample_recon_loss.item())
    test_kl_loss_over_time.append(test_sample_kl_loss.item())

for sample in JB_train_dataset:
    train_sample_name = sample['name']
    
    train_sample_image = sample['image']
    
    
    train_sample_image = torch.unsqueeze(train_sample_image,0)
    
    # # # forward pass the images through the network

    train_sample_image_recon,train_sample_z_vector,train_sample_mu_z,train_sample_log_var_z = best_model(train_sample_image)
            
    
    # Calculate loss
    train_sample_total_loss,train_sample_recon_loss,train_sample_kl_loss = vae_loss(train_sample_image_recon, train_sample_image, train_sample_mu_z, train_sample_log_var_z,variational_beta)
    
    train_test_loss_over_time.append(train_sample_total_loss.item())
    train_test_recon_loss_over_time.append(train_sample_recon_loss.item())
    train_test_kl_loss_over_time.append(train_sample_kl_loss.item())
    latent_space_training_set.append(train_sample_mu_z.detach().numpy())
    latent_space_array = np.squeeze(np.array(latent_space_training_set))

# write evaluation results to file
write_list_to_file(train_test_loss_over_time, test_results_path+ '/'+ 'Train_loss_eval.csv')
write_list_to_file(test_loss_over_time, test_results_path+ '/'+ 'Test_loss_eval.csv')  
'''
#%% Evaluate loss of trained model on test and train datasets

best_model = best_model.to(device) # trying something new
best_model.eval()

#initialize arrays for storing loss on training and test datasets
train_test_loss_over_time = np.zeros(len(JB_train_dataset))
train_test_recon_loss_over_time = np.zeros(len(JB_train_dataset))
train_test_kl_loss_over_time = np.zeros(len(JB_train_dataset))



test_loss_over_time = np.zeros(len(JB_test_dataset))
test_recon_loss_over_time = np.zeros(len(JB_test_dataset))
test_kl_loss_over_time = np.zeros(len(JB_test_dataset))

#initialize arrays for storing loss on target dataset
target_loss_over_time = np.zeros(len(target_dataset))
target_recon_loss_over_time = np.zeros(len(target_dataset))
target_kl_loss_over_time = np.zeros(len(target_dataset))



latent_space_training_set = np.zeros((len(JB_train_dataset),int(config['model_params']['latent_dim'])))

for test_sample_index,test_sample in enumerate(JB_test_dataset):
    test_sample_name = test_sample['name']
    
    test_sample_image = test_sample['image']
    
    
    test_sample_image = torch.unsqueeze(test_sample_image,0)
    
    test_sample_image = test_sample_image.to(device) # Trying something new
    
    # # # forward pass the images through the network

    test_sample_image_recon,test_sample_z_vector,test_sample_mu_z,test_sample_log_var_z = best_model(test_sample_image)
            
    # Calculate loss
    test_sample_total_loss,test_sample_recon_loss,test_sample_kl_loss = vae_loss(test_sample_image_recon, test_sample_image, test_sample_mu_z, test_sample_log_var_z,variational_beta)
    
    test_loss_over_time[test_sample_index] = test_sample_total_loss.item()
    test_recon_loss_over_time[test_sample_index]=test_sample_recon_loss.item()
    test_kl_loss_over_time[test_sample_index] = test_sample_kl_loss.item()
    
    

for train_idx,sample in enumerate(JB_train_dataset):
    train_sample_name = sample['name']
    
    train_sample_image = sample['image']
    
    
    train_sample_image = torch.unsqueeze(train_sample_image,0)
    train_sample_image = train_sample_image.to(device) # trying something new
    
    # # # forward pass the images through the network

    train_sample_image_recon,train_sample_z_vector,train_sample_mu_z,train_sample_log_var_z = best_model(train_sample_image)
            
    
    # Calculate loss
    train_sample_total_loss,train_sample_recon_loss,train_sample_kl_loss = vae_loss(train_sample_image_recon, train_sample_image, train_sample_mu_z, train_sample_log_var_z,variational_beta)
    
    train_test_loss_over_time[train_idx] = train_sample_total_loss.item()
    train_test_recon_loss_over_time[train_idx] = train_sample_recon_loss.item()
    train_test_kl_loss_over_time[train_idx] = train_sample_kl_loss.item()
    #latent_space_training_set[train_idx,:] = train_sample_mu_z.detach().numpy() # original
    
    #Trying something new - copy to cpu first before converting to numpy
    
    latent_space_training_set[train_idx,:] = train_sample_mu_z.detach().cpu().numpy()
    
# =============================================================================
#     train_test_loss_over_time.append(train_sample_total_loss.item())
#     train_test_recon_loss_over_time.append(train_sample_recon_loss.item())
#     train_test_kl_loss_over_time.append(train_sample_kl_loss.item())
#     latent_space_training_set.append(train_sample_mu_z.detach().numpy())
# =============================================================================
    latent_space_array = np.squeeze(np.array(latent_space_training_set))

# test trained model on target
for target_idx,sample in enumerate(target_dataset):
    target_sample_name = sample['name']
    
    target_sample_image = sample['image']
    
    
    target_sample_image = torch.unsqueeze(target_sample_image,0)
    target_sample_image = target_sample_image.to(device) # trying something new
    
    # # # forward pass the images through the network

    target_sample_image_recon,target_sample_z_vector,target_sample_mu_z,target_sample_log_var_z = best_model(target_sample_image)
            
    
    # Calculate loss
    target_sample_total_loss,target_sample_recon_loss,target_sample_kl_loss = vae_loss(target_sample_image_recon, target_sample_image, target_sample_mu_z, target_sample_log_var_z,variational_beta)
    
    target_loss_over_time[target_idx] = target_sample_total_loss.item()
    target_recon_loss_over_time[target_idx] = target_sample_recon_loss.item()
    target_kl_loss_over_time[target_idx] = target_sample_kl_loss.item()


#compute statistics of loss of the model on test dataset 
eval_mean_test_loss = np.mean(test_recon_loss_over_time)
eval_std_test_loss = np.std(test_recon_loss_over_time)
run["metrics/mean_test_loss"].log(eval_mean_test_loss)
run["metrics/std_test_loss"].log(eval_std_test_loss)


#compute statistics of loss of the model on target dataset 
eval_mean_target_loss = np.mean(target_recon_loss_over_time)
eval_std_target_loss = np.std(target_recon_loss_over_time)
run["metrics/mean_target_loss"].log(eval_mean_target_loss)
run["metrics/std_target_loss"].log(eval_std_target_loss)



# write evaluation results to file
write_list_to_file(train_test_loss_over_time, test_results_path+ '/'+ 'Train_loss_eval.csv')
write_list_to_file(test_loss_over_time, test_results_path+ '/'+ 'Test_loss_eval.csv')  



#%% Plot eval metrics - evaluate reconstruction error on training and testing datasets
eval_loss_test_dataset=plt.figure()
plt.title('Loss on test dataset')
plt.plot(test_loss_over_time,label='Total loss')
plt.plot(test_recon_loss_over_time,label='reconstruction loss')
plt.plot(test_kl_loss_over_time,label='kl loss')
plt.plot(target_loss_over_time,label='target recon loss')
plt.legend()
plt.grid()
plt.show(block=False)

eval_loss_train_dataset=plt.figure()
plt.title('Loss on train dataset')
plt.plot(train_test_loss_over_time,label='Total loss')
plt.plot(train_test_recon_loss_over_time,label='reconstruction loss')
plt.plot(train_test_kl_loss_over_time,label='kl loss')
plt.plot(target_loss_over_time,label='target recon loss')
plt.legend()
plt.grid()
plt.show(block=False)

binwidth=500
loss_distb_fig = plt.figure()
plt.title('loss distributions')
plt.hist(train_test_loss_over_time,density=True, alpha=0.5, bins=np.arange(min(train_test_loss_over_time), max(train_test_loss_over_time) + binwidth, binwidth),label='training dataset')
plt.hist(test_loss_over_time,density=True, alpha=0.5, bins=np.arange(min(test_loss_over_time), max(test_loss_over_time) + binwidth, binwidth),label='test dataset')
plt.legend()
plt.grid()
plt.show(block=False)


test_eval_images_export = visualize_vae_output_eval(test_sample_image, test_sample_image_recon,test_sample_total_loss,test_sample_recon_loss,test_sample_kl_loss)

train_eval_images_export = visualize_vae_output_eval(train_sample_image, train_sample_image_recon,train_sample_total_loss,train_sample_recon_loss,train_sample_kl_loss)

target_eval_images_export = visualize_vae_output_eval(target_sample_image, target_sample_image_recon,target_sample_total_loss,target_sample_recon_loss,target_sample_kl_loss)

run["predictions/loss_on_train_dataset"].upload(File.as_image(eval_loss_train_dataset))
run["predictions/loss_on_test_dataset"].upload(File.as_image(eval_loss_test_dataset))

run["predictions/test_eval_recon_imgs"].upload(File.as_image(test_eval_images_export))
run["predictions/train_eval_recon_imgs"].upload(File.as_image(train_eval_images_export))
run["predictions/target_eval_recon_imgs"].upload(File.as_image(target_eval_images_export))

run["metrics/loss_distb"].upload(File.as_image(loss_distb_fig))
#[CONTINUE FROM HERE]

# Save this for later - tensorboard visualizations for autoencoders (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html)  


#%% Standardize latent space
from sklearn.preprocessing import StandardScaler,MinMaxScaler
std_latent_space_array = StandardScaler().fit_transform(latent_space_array)


#%% Dimensionality reduction

from sklearn import manifold
from numpy.random import RandomState


import seaborn as sns

rng = RandomState(0)
t_sne = manifold.TSNE(
    n_components=2,learning_rate=200,verbose=1,
    perplexity=30,
    n_iter=1000,
    init="pca",
    random_state=24,metric="cosine")
reduced_latent_space = t_sne.fit_transform(std_latent_space_array)

scatterplot_figure = plt.figure()
plt.scatter(reduced_latent_space[:,0],reduced_latent_space[:,1])
plt.grid()
plt.show(block=False)

run["predictions/latent_space"].upload(File.as_image(scatterplot_figure)) 

plt.figure()
plt.scatter(reduced_latent_space[:,0],reduced_latent_space[:,1])
plt.grid()
plt.show(block=False)





from sklearn.decomposition import PCA


pca = PCA(n_components=2,whiten=False)
pca_components = pca.fit_transform(std_latent_space_array)

#pca_components = pca.components_.T
pca_fig = plt.figure()
#plot pca and tsne spaces overlaid

plt.scatter(reduced_latent_space[:,0],reduced_latent_space[:,1],label='TSNE reduction',c='red',alpha=0.4)
plt.scatter(pca_components[:,0],pca_components[:,1],label='PCA reduction',c='blue')
plt.grid()
plt.legend()
plt.show(block=False)

plt.figure()
plt.scatter(pca_components[:,0],pca_components[:,1],alpha=0.1)
plt.grid()
plt.show(block=False)



# Normalize the TSNE and PCA embeddings

norm_pca_components = MinMaxScaler().fit_transform(pca_components)
norm_reduced_latent_space = MinMaxScaler().fit_transform(reduced_latent_space)

reduced_latent_space_figure = plt.figure(figsize = (15,10))
plt.subplot(121)



plt.scatter(norm_reduced_latent_space[:,0],norm_reduced_latent_space[:,1],label='TSNE reduction',c='red',alpha=0.2)
plt.grid()
plt.legend()

plt.subplot(122)
plt.scatter(norm_pca_components[:,0],norm_pca_components[:,1],label='PCA reduction',c='blue',alpha=0.2)
plt.grid()
plt.legend()
plt.show(block=False)

run["predictions/pca_latent_space"].upload(File.as_image(reduced_latent_space_figure)) 

run.stop()