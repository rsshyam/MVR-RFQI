sh scripts/run_seq_env_script.sh --env 'Pendulum-v1' --type 'rfqi' --data_eps 0.3 --rho 0.1 --gendata_pol 'sac' --dtalrnt 'False' --max_trn_steps 50000 --batch_size 100 --comment '50000-100' 

#training will store the model in models folder and evaluation will call upon this saved model. 


#this script will automatically run the rfqi training and evaluation for pendulum with the mentioned hyperparameters
#Also rho can take multiple values separated by commas and the training and evaluation will run sequentially for these rhos
#change type to 'fqi' for FQI training and evaluation