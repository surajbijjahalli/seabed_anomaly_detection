#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:56:56 2022

@author: surajb
"""

# Plot performance curves across multiple models - plot using csv file from neptune

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/surajb/seabed_anomaly_detection_saved_models/vae_result_summary.csv') #autoencoder_results_3.csv is the summary file for vanilla autoencoder results
#data.plot(data['parameters/model_params/latent_dim'],data['metrics/recon_error_separation(last)'])
#data.plot(data.parameters/model params/latent_dim,data.metrics/recon error separation)
#something = data['parameters/model_params/latent_dim']
#something_else = data['metrics/recon_error_separation (last)']



#plt.figure()
#plt.plot(data['parameters/model_params/latent_dim'],data['metrics/recon_error_separation (average)'])
#plt.show()

plt.figure()
plt.plot(data['parameters/model_params/latent_dim'],data['metrics/mean_target_loss (average)'],"o",linestyle="dashdot",color='r',label='Outlier')
plt.xlabel("d",style='italic')
plt.ylabel("MSE",style='italic')
#plt.plot(data['parameters/model_params/latent_dim'], data['metrics/mean_target_loss (average)'] - data['metrics/std_target_loss (average)'], color='r', linestyle='dashed')
#plt.plot(data['parameters/model_params/latent_dim'], data['metrics/mean_target_loss (average)'] + data['metrics/std_target_loss (average)'], color='r', linestyle='dashed')

#filler = plt.fill_between(data['parameters/model_params/latent_dim'], data['metrics/mean_target_loss (average)'] - data['metrics/std_target_loss (average)'], data['metrics/mean_target_loss (average)'] + data['metrics/std_target_loss (average)'], color='r', alpha=0.2)



plt.plot(data['parameters/model_params/latent_dim'],data['metrics/mean_test_loss (last)'],"^",linestyle="dashed",color='b',label='Inlier')
#plt.plot(data['parameters/model_params/latent_dim'], data['metrics/mean_test_loss (last)'] - data['metrics/std_test_loss (last)'], color='b', linestyle='dashed')
#plt.plot(data['parameters/model_params/latent_dim'], data['metrics/mean_test_loss (last)'] + data['metrics/std_test_loss (last)'], color='b', linestyle='dashed')

#blue_filler=plt.fill_between(data['parameters/model_params/latent_dim'], data['metrics/mean_test_loss (last)'] - data['metrics/std_test_loss (last)'], data['metrics/mean_test_loss (last)'] + data['metrics/std_test_loss (last)'], color='b', alpha=0.2)



plt.grid()
plt.legend()

plt.show()
