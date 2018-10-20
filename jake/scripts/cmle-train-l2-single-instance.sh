#!/bin/bash
echo "Submitting job to train ML model on GCP ML Engine"

TRIAL="6"
GCS_DIR="gs://trash_net"
JOB_NAME=trashNet_${VERSION}_trial_${TRIAL} #update trail number with every run
REGION=us-central1
RUNTIME_VERSION=1.6
VERSION="v1"
PACKAGE_PATH=trainer
MODEL_NAME=trashNet_${VERSION}
MODEL_DIR=${GCS_DIR}/trained_models/${MODEL_NAME}
SRC_DATA_PATH=${GCS_DIR}/dataset
IMG_WIDTH_PIXEL=160
IMG_HEIGHT_PIXEL=160
IMG_CHANNELS=3
TEST_SIZE=0.2


gcloud ml-engine jobs submit training $JOB_NAME \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        --region=${REGION} \
        --config=${CONFIG=config.yaml} \
        --runtime-version=${RUNTIME_VERSION} \
        --job-dir=${MODEL_DIR} \
        -- \
        --src-data-path=${SRC_DATA_PATH} \
        --img-width-pixel=${IMG_WIDTH_PIXEL} \
        --img-height-pixel=${IMG_HEIGHT_PIXEL} \
        --img-channels=${IMG_CHANNELS} \
        --test-size=${TEST_SIZE} \
        --train-batch-size=128 \
        --test-batch-size=128 \
        --log-step-count-steps=1 \
        --save-checkpoints-secs=300 \
        --keep-checkpoint-max=3 \
        --learning-rate=0.0001 \
        --first-layer-size=100 \
        --num-layers=4 \
        --layer-sizes-scale-factor=0.9 \
        --dropout-rate=0.7 \
        --num-epochs=10 \
        --checkpoint-epochs=5 \
        --validation-split=0.2 \
        --threshold=0.3 \
        --verbosity="INFO"
