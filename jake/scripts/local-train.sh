#!/bin/bash
echo "Training ML model locally, sourcing dataset locally"

# set the tf-hub module cache directory
export TFHUB_CACHE_DIR=./tfhub_cache

PACKAGE_PATH=trainer
VERSION="v1"
MODEL_NAME=vanilla_GAN_hinge_loss_${VERSION}
MODEL_DIR=trained_models/${MODEL_NAME}
SRC_IMG_DIR=dataset/img_align_celeba
SRC_TFRECORD_PATH=dataset/celeba_dataset
IMG_WIDTH_PIXEL=160
IMG_HEIGHT_PIXEL=160
IMG_CHANNELS=3
TEST_SIZE=0.2
Z_SIZE=128
SUMMARY_PER_N_STEPS=20
SAVE_MODEL_PER_N_STEPS=100
MAX_TRAINING_STEPS=20000


gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        --job-dir=${MODEL_DIR} \
        -- \
        --src-img-dir=${SRC_IMG_DIR} \
        --src-tfrecord-path=${SRC_TFRECORD_PATH} \
        --img-width-pixel=${IMG_WIDTH_PIXEL} \
        --img-height-pixel=${IMG_HEIGHT_PIXEL} \
        --img-channels=${IMG_CHANNELS} \
        --z-size=${Z_SIZE} \
        --summary-per-n-steps=${SUMMARY_PER_N_STEPS} \
        --save-per-n-steps=${SAVE_MODEL_PER_N_STEPS} \
        --max-train-steps=${MAX_TRAINING_STEPS} \
        --g-learn-rate=0.0001 \
        --d-learn-rate=0.0001 \
        --beta1=0.0 \
        --beta2=0.9 \
        --epochs=100 \
        --shuffle-buffer=512 \
        --batch-size=64 \
        --verbosity="INFO" \
        --reuse-job-dir


# ls ${MODEL_DIR}/export/estimator
# MODEL_LOCATION=${MODEL_DIR}/export/estimator/$(ls ${MODEL_DIR}/export/estimator | tail -1)
# echo ${MODEL_LOCATION}
# ls ${MODEL_LOCATION}

# # invoke trained model to make prediction given new data instances
# gcloud ml-engine local predict --model-dir=${MODEL_LOCATION} --json-instances=data/new-data.json

