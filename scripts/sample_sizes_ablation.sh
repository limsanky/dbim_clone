export PYTHONPATH=$PYTHONPATH:./

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
export CUDA_VISIBLE_DEVICES=4,5,6,7
run_args="--nproc_per_node 4 \
          --master_port 29511"

# Batch size per GPU
BS=64

# Dataset and checkpoint
DATASET_NAME=f2c256

SPLIT=train

source /root/code/dbim_clone/scripts/args.sh $DATASET_NAME

UNET=cbm_unet_sizeS
NUM_CH=256
NUM_RES_BLOCKS=1
ATTN=16,32

IMG_SIZE=256

# Number of function evaluations (NFE)
N=5
NFE=$((N-1))

# Sampler
GEN_SAMPLER=dbim

USE_FP16=True

MODEL_PATH=/root/code/ddbm_org/samplings/ablations/dbm_bms/ema_0.9993_140000.pt

torchrun $run_args /root/code/dbim_clone/sample.py --steps $NFE --sampler $GEN_SAMPLER --batch_size $BS \
 --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
 --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
 --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
 --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT\
 ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \
 ${ETA:+ --eta="${ETA}"} \
 ${ORDER:+ --order="${ORDER}"} --use_fp16 $USE_FP16 --attention_resolutions $ATTN --unet_type $UNET
