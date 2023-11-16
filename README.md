
# marine_anomaly_detection
Detecting anomalies in seafloor images. Train a Variational Autoencoder (VAE) to reconstruct images of the seafloor. Anomalies in large marine datasets are detected by monitoring the reconstruction error and the density of the latent features. The key idea is that anomalous images will have a high reconstruction error and will lie in the low density regions of the latent feature space. The Arxiv manuscript is here: https://arxiv.org/abs/2306.04834 ![anomaly_system_overview](/assets/anomaly_system_overview.png)

## Example of high reconstruction error for anomalous images (images containing artificial objects)
![contoured_anomaly_maps](/assets/contoured_anomaly_maps.png)

## Example of anomalous images in the low-density regions of the latent feature space
![latent_space_plot_cluster](/assets/kde_plot_testdata.png)

## Usage
* Create a yaml config file in the config folder. A baseline configuration is provided as an example. 
* To train a VAE, run Main.py with the desired config file. For example, to train a model with the baseline configuration parameters:
  * `python3 Main.py baseline.yaml`
* Several config files can be created with different latent dimensions. To create these files and place them in the config folder:
  * `./create_config_files.sh` . The script modifies the baseline.yaml file, creates a copy and places it in the config folder.
* The Main program can be executed repeatedly on different config files in the config folder. After creating config files for all your required experiments, simply run:
  * `./run_program_iterate.sh` 

