#!/usr/bin/env python
import os
import argparse
from datetime import datetime
import tensorflow as tf
import json
import glob
import numpy as np
import trainer.model
import pandas as pd
# import trainer.helper as helper
from keras.callbacks import EarlyStopping, TensorBoard, Callback, ModelCheckpoint
from keras.models import load_model
from tensorflow.python.lib.io import file_io
# 
tf.enable_eager_execution()


def initialise_hyper_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable
    """

    #data source params
    #--------------------------------------------
    args_parser.add_argument(
        '--src-img-dir', required=True, type=str,
        default='dataset/img_align_celeba',
        help='Source image dataset directory.'
    )

    args_parser.add_argument(
        '--src-tfrecord-path', required=True, type=str,
        default='dataset/celeba_dataset',
        help='Source tfRecord dataset filepath.'
    )

    args_parser.add_argument(
        '--img-width-pixel', required=True, type=int,
        default=128, help='Width of images (after rescaling) in pixels.'
    )

    args_parser.add_argument(
        '--img-height-pixel', required=True, type=int, default=128, 
        help='Height of images (after rescaling) in pixels.'
    )

    args_parser.add_argument(
        '--img-channels', required=True, type=int, default=3, 
        help='Number of channels of images.'
    )

    args_parser.add_argument(
        '--z-size', required=True, type=int, default=128, 
        help='Dimension size of the input vector into generator.'
    )
    
    args_parser.add_argument(
        '--summary-per-n-steps', required=True, type=int, default=20, 
        help='write summary after every N batch'
    )

    args_parser.add_argument(
        '--save-per-n-steps', required=True, type=int, default=100, 
        help='save model every N batch'
    )

    args_parser.add_argument(
        '--max-train-steps', required=True, type=int, default=20000, 
        help='maximum number of batches to train for'
    )

    #experiment configs
    #--------------------------------------------
    args_parser.add_argument(
        '--job-dir', required=True, default='results',
        type=str, help='dir-path to write checkpoints and export model'
    )

    #training configs
    #--------------------------------------------
    args_parser.add_argument(
        '--g-learn-rate', type=float, default=0.0001,
        help='Generator learning rate'
    )

    args_parser.add_argument(
        '--d-learn-rate', type=float, default=0.0004,
        help='Discriminator learning rate'
    )

    args_parser.add_argument(
        '--alpha', type=float, default=0.2,
        help='Alpha parameter for the leaky ReLu'
    )

    args_parser.add_argument(
        '--beta1', type=float, default=0.0,
        help='Adam optimizer beta1'
    )

    args_parser.add_argument(
        '--beta2', type=float, default=0.9,
        help='Adam optimizer beta2'
    )

    args_parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of epochs to train'
    )

    args_parser.add_argument(
        '--shuffle-buffer', type=int, default=512,
        help='number of examples in buffer to shuffle'
    )

    args_parser.add_argument(
        '--batch-size', type=int, default=64,
        help='number of training examples in a batch'
    )

    args_parser.add_argument(
        '--checkpoint-epochs', type=int, default=2,
        help='Checkpoint per n training epochs'
    )

    args_parser.add_argument(
        '--verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO'
    )

    args_parser.add_argument(
        '--reuse-job-dir', action='store_true',
        help="""Flag to decide if the model checkpoint should be re-used from the job-dir. 
          If False then the job-dir will be deleted"""
    )

    parsed_args, unknown = args_parser.parse_known_args()

    return parsed_args



# ******************************************************************************
# THIS IS ENTRY POINT FOR THE TRAINER TASK
# ******************************************************************************
def main():

    # Set python level verbosity
    tf.logging.set_verbosity(HYPER_PARAMS.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[HYPER_PARAMS.verbosity] / 10)
    print("")
    print('Hyper-parameters:')
    print(".......................................")
    print(HYPER_PARAMS)
    print("")
    # If job_dir_reuse is False then remove the job_dir if it exists
    print("Resume training:", HYPER_PARAMS.reuse_job_dir)
    print(".......................................")
    if not HYPER_PARAMS.reuse_job_dir:

        if tf.gfile.Exists(HYPER_PARAMS.job_dir):
            tf.gfile.DeleteRecursively(HYPER_PARAMS.job_dir)
            print("Deleted job_dir {} to avoid re-use".format(HYPER_PARAMS.job_dir))
        else:
            print("No job_dir available to delete")
    else:
        print("Reusing job_dir {} if it exists".format(HYPER_PARAMS.job_dir))  

    # Instantiate model object  
    print("")
    print("Instantiating model obj & loading necessary files:")
    print(".......................................")
    model = trainer.model.Model()

    model.train()
    # # Run training and evaluation operations
    # model.run_experiment()
    # # Save model
    # model.to_savedmodel(HYPER_PARAMS.job_dir)


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'



if __name__ == "__main__":
  
    main()
