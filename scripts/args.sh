DATASET_NAME=$1

UNET=adm
ATTN=8,16,32

if [[ $DATASET_NAME == "e2h" ]]; then
    DATA_DIR=/root/data/edges2handbags/
    DATASET=edges2handbags
    IMG_SIZE=64

    NUM_CH=192
    NUM_RES_BLOCKS=3
    ATTN_TYPE=True

    EXP="e2h${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="vp"
elif [[ $DATASET_NAME == "diode" ]]; then
    DATA_DIR=/root/data/diode/diode-normal-256/
    DATASET=diode
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True
    # ATTN_TYPE=flash

    EXP="diode${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="vp"
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    DATA_DIR=/root/data/imagenet/
    DATASET=imagenet_inpaint_center
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=False

    EXP="imagenet_inpaint_center${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=16
    DROPOUT=0
    CLASS_COND=True

    PRED="i2sb_cond"
elif [[ $DATASET_NAME == "f2c64" ]]; then
    DATA_DIR=/root/data/face2comics/resized_64/
    DATASET=f2c
    IMG_SIZE=64

    NUM_CH=128
    NUM_RES_BLOCKS=3
    ATTN_TYPE=True
    ATTN=4,8
    UNET=cbm_unet

    EXP="f2c${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="ve"
elif [[ $DATASET_NAME == "f2c128" ]]; then
    DATA_DIR=/root/data/face2comics/resized_128/
    DATASET=f2c
    IMG_SIZE=128

    NUM_CH=192
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True
    ATTN=8,16
    UNET=cbm_unet

    EXP="f2c${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="ve"
elif [[ $DATASET_NAME == "f2c256" ]]; then
    DATA_DIR=/root/data/face2comics/
    DATASET=f2c
    IMG_SIZE=256

    NUM_CH=256
    NUM_RES_BLOCKS=2
    ATTN_TYPE=True
    ATTN=16,32
    UNET=cbm_unet

    EXP="f2c${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=20000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False

    PRED="ve"
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
    SIGMA_MAX=80.0
    SIGMA_MIN=0.002
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "i2sb_cond" ]]; then
    EXP+="_i2sb_cond"
    COND=concat
    BETA_MAX=1.0
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi