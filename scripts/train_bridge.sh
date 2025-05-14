export PYTHONPATH=$PYTHONPATH:./

DATASET_NAME=f2c64
TRAIN_MODE=i2sb

source scripts/args.sh $DATASET_NAME

FREQ_SAVE_ITER=5000
EXP=${DATASET_NAME}-${TRAIN_MODE}

# CKPT=assets/ckpts/256x256_diffusion_fixedsigma.pt

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# run_args="--nproc_per_node 8 \
#           --master_port 29511"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
run_args="--nproc_per_node 4 \
          --master_port 29511"

MICRO_BS=64

PRED="i2sb_cond"

torchrun $run_args train.py --exp=$EXP \
 --class_cond $CLASS_COND  \
 --dropout $DROPOUT  --microbatch $MICRO_BS \
 --image_size $IMG_SIZE  --num_channels $NUM_CH  \
 --num_res_blocks $NUM_RES_BLOCKS  --condition_mode=$COND  \
 --noise_schedule=$PRED    \
 --use_new_attention_order $ATTN_TYPE  \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET  \
 --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN  \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
 ${CKPT:+ --resume_checkpoint="${CKPT}"} 