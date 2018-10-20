#!/bin/bash
MODEL_NAME="doc2catL2"
REGION=us-central1
DEPLOYMENT_SOURCE="gs://hubba_ml/productClassifiers/productClassifier/trained_models/doc2cat_l2_v1/export"
DEPLOYMENT_CONFIG=deploy.yaml
VERSION=v1
RUNTIME_VERSION=1.4
DEPLOYMENT_TEST_FILE=online_prediction_test.json

echo "Deploying model to ML Engine"
#create model
gcloud ml-engine models create $MODEL_NAME --regions $REGION
#create model version & deploy from model export directory
gcloud ml-engine versions create $VERSION --model $MODEL_NAME --origin $DEPLOYMENT_SOURCE --config $DEPLOYMENT_CONFIG --runtime-version $RUNTIME_VERSION
#check model deployment
gcloud ml-engine versions describe $VERSION --model $MODEL_NAME
#check for model online inference
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION --json-instances $DEPLOYMENT_TEST_FILE