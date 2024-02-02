#!/bin/bash

# Set default values for train_fqi.py
DATA_EPS=1
ENV="pilcocartpole-v0"
MAX_TRN_STEPS=2000
EVAL_FREQ=50
DEVICE="cuda"
DATA_SIZE=1000000
BATCH_SIZE=100
GENDATA_POL="sac"
COMMENT="2000-100"
TYPE="rfqi"
EVAL_EPISODES=20
DTALRNT='True'
NSAMPLES=3


# Parse command line arguments for train_fqi.py
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --data_eps)
        DATA_EPS="$2"
        shift
        shift
        ;;
        --env)
        ENV="$2"
        shift
        shift
        ;;
        --max_trn_steps)
        MAX_TRN_STEPS="$2"
        shift
        shift
        ;;
        --eval_freq)
        EVAL_FREQ="$2"
        shift
        shift
        ;;
        --device)
        DEVICE="$2"
        shift
        shift
        ;;
        --data_size)
        DATA_SIZE="$2"
        shift
        shift
        ;;
        --batch_size)
        BATCH_SIZE="$2"
        shift
        shift
        ;;
        --gendata_pol)
        GENDATA_POL="$2"
        shift
        shift
        ;;
        --comment)
        COMMENT="$2"
        shift
        shift
        ;;
        --type)
        TYPE="$2"
        shift
        shift
        ;;
        --eval_episodes)
        EVAL_EPISODES="$2"
        shift
        shift
        ;;
        --nsamples)
        NSAMPLES="$2"
        shift
        shift
        ;;
        --dtalrnt)
        DTALRNT="$2"
        shift
        shift
        ;;
        --rho)
        # Accept comma-separated string of rho values and convert to array
        IFS=',' read -ra RHO_VALUES <<< "$2"
        shift
        ;;
        *)
        shift
        ;;
    esac
done

echo "DATA_EPS = $DATA_EPS"
echo "COMMENT = $COMMENT"
echo "env=$ENV"
echo "max_trn_steps=$MAX_TRN_STEPS"
echo "eval_freq=$EVAL_FREQ"
echo "device=$DEVICE"
echo "data_size=$DATA_SIZE"
echo "batch_size=$BATCH_SIZE"
echo "gendata_pol=$GENDATA_POL"
echo "type=$TYPE"
echo "eval_episodes=$EVAL_EPISODES"
echo "dtlrnt=$DTALRNT"
echo "nsamples=$NSAMPLES"
echo "rho=${RHO_VALUES[@]}"


for rho in "${RHO_VALUES[@]}"
do
    if [ "$TYPE" == "fqi" ]; then
        # Run train_fqi.py script
        python train_fqi.py \
        --data_eps="$DATA_EPS" \
        --env="$ENV" \
        --max_trn_steps="$MAX_TRN_STEPS" \
        --eval_freq="$EVAL_FREQ" \
        --device="$DEVICE" \
        --data_size="$DATA_SIZE" \
        --batch_size="$BATCH_SIZE" \
        --gendata_pol="$GENDATA_POL" \
        --comment="$COMMENT" \
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"

        # Run eval_fqi.py script
        python eval_fqi.py \
        --data_eps="$DATA_EPS" \
        --gendata_pol="$GENDATA_POL" \
        --env="$ENV" \
        --eval_episodes="$EVAL_EPISODES" \
        --comment="$COMMENT" \
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"
    elif [ "$TYPE" == "rfqi" ]; then
        # Run train_fqi.py script
        python train_rfqi.py \
        --data_eps="$DATA_EPS" \
        --env="$ENV" \
        --max_trn_steps="$MAX_TRN_STEPS" \
        --eval_freq="$EVAL_FREQ" \
        --device="$DEVICE" \
        --data_size="$DATA_SIZE" \
        --batch_size="$BATCH_SIZE" \
        --gendata_pol="$GENDATA_POL" \
        --rho="$rho" \
        --comment="$COMMENT" \
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"

    # Run eval_fqi.py script
        python eval_rfqi.py \
        --data_eps="$DATA_EPS" \
        --gendata_pol="$GENDATA_POL" \
        --env="$ENV" \
        --eval_episodes="$EVAL_EPISODES" \
        --rho="$rho" \
        --comment="$COMMENT" \
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"
    elif [ "$TYPE" == "rfqieval" ]; then

        # Run eval_fqi.py script
        python eval_rfqi.py \
        --data_eps="$DATA_EPS" \
        --gendata_pol="$GENDATA_POL" \
        --env="$ENV" \
        --eval_episodes="$EVAL_EPISODES" \
        --rho="$rho" \
        --comment="$COMMENT"\
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"
    elif [ "$TYPE" == "fqieval" ]; then

        # Run eval_fqi.py script
        python eval_fqi.py \
        --data_eps="$DATA_EPS" \
        --gendata_pol="$GENDATA_POL" \
        --env="$ENV" \
        --eval_episodes="$EVAL_EPISODES" \
        --comment="$COMMENT"\
        --dtalrnt="$DTALRNT" \
        --nsamples="$NSAMPLES"
    fi
done