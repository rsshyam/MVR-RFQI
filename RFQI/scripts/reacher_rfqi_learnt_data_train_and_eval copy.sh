sh scripts/run_seq_env_script.sh --env 'bacreacher-v0' --type 'rfqi' --data_eps 0.3 --rho 0.1 --nsamples 2000 --dtalrnt 'True' --gendata_pol 'torch_sac' --max_trn_steps 40000 --batch_size 500 --comment '40000-500-500' --eval_freq 500

#training will store the model in models folder and evaluation will call upon this saved model. 


#this script will automatically run the rfqi training and evaluation for pendulum with the mentioned hyperparameters
#Also rho can take multiple values separated by commas and the training and evaluation will run sequentially for these rhos
#change type to 'fqi' for FQI training and evaluation

#For reacher we specifically focus on perturbations in joint_stiffness_x w.r.t. changes in springref parameter. This evaluation is done below

python eval_rfqi_springref_check.py --data_eps=0.3 --gendata_pol=torch_sac --env=bacreacher-v0 --eval_episodes=20 --rho=0.1 --comment=40000-500-500 --dtalrnt=True --nsamples=2000 --springref=100

python eval_fqi_springref_check.py --data_eps=0.3 --gendata_pol=torch_sac --env=bacreacher-v0 --eval_episodes=20 --rho=0.1 --comment=40000-500-500 --dtalrnt=True --nsamples=2000 --springref=100