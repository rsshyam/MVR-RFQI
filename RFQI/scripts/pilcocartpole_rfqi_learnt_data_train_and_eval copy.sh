sh scripts/run_seq_env_script.sh --env 'pilcocartpole-v0' --type 'rfqi' --data_eps 0.3 --rho 0.3 --gendata_pol 'mpc' --nsamples 150 --max_trn_steps 5000 --batch_size 100 --comment '5000-100' 

#training will store the model in models folder and evaluation will call upon this saved model. 


#this script will automatically run the rfqi training and evaluation for pendulum with the mentioned hyperparameters
#Also rho can take multiple values separated by commas and the training and evaluation will run sequentially for these rhos
#change type to 'fqi' for FQI training and evaluation