#! /bin/bash

#echo "Hello"

#sleep 3

#echo "Have a nice day"

#echo "Bash version ${BASH_VERSION}..."
#for i in {0..10..2}
#do
#  echo "Welcome $i times"
#done

#initialize experiment counter
exp_count=0
cd config
latent_dim=(128 256 512 1024)
beta=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 4 6 8)

for dim in ${latent_dim[@]}
  do 
    for value in ${beta[@]}
  
      do 
      
        exp_count=$((exp_count+1))
        # navigate to config directory
        # replace existing experiment name with exp_count
        cat baseline.yaml | sed -e "s#experiment_name: 'baseline'#experiment_name: '$exp_count'#g" -e "s#latent_dim: 256#latent_dim: $dim#g" -e "s#kld_weight: 0.01#kld_weight: $value#g" > $exp_count.yaml
        
        # save as new config file with name exp_count
        # run Main.py on newly created config file
        
        # test actions
        #echo $exp_count  
        #echo $dim $value
    done    
done

#echo latent_dim[1]
