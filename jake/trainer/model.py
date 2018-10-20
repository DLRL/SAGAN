#!/usr/bin/env python
import trainer.task
from trainer.prepare_data import make_tfrecord
from trainer.input_datapipe import create_tfdataset
import numpy as np
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
#
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
# 
from candidate_models.model_vanilla_gan import Generator, Discriminator

tfe = tf.contrib.eager


class Model():

    def __init__(self):
        # specify model directory
        self.model_dir = trainer.task.HYPER_PARAMS.job_dir
        self.log_dir = os.path.join(self.model_dir, 'logs')
        # alpha param for leaky ReLu
        self.alpha = trainer.task.HYPER_PARAMS.alpha
        # learning rates
        self.g_lr = trainer.task.HYPER_PARAMS.g_learn_rate
        self.d_lr = trainer.task.HYPER_PARAMS.d_learn_rate
        # beta params for the Adam optimizer
        self.beta1 = trainer.task.HYPER_PARAMS.beta1
        self.beta2 = trainer.task.HYPER_PARAMS.beta2
        # training batch size & epoch
        self.batch_size = trainer.task.HYPER_PARAMS.batch_size
        self.epochs = trainer.task.HYPER_PARAMS.epochs
        # dimension of random vector (input to generator)
        self.z_size = trainer.task.HYPER_PARAMS.z_size
        # record every X intervals and max training steps
        self.summary_per_n_steps = trainer.task.HYPER_PARAMS.summary_per_n_steps
        self.max_train_steps = trainer.task.HYPER_PARAMS.max_train_steps
        # optimizer learning rate
        self.g_learn_rate = trainer.task.HYPER_PARAMS.g_learn_rate
        self.d_learn_rate = trainer.task.HYPER_PARAMS.d_learn_rate
        
        # make tfRecords from source image directory if it does not already exist
        if not os.path.isfile(trainer.task.HYPER_PARAMS.src_tfrecord_path+'.tfrecord'):
            make_tfrecord(img_dir='{}/*.jpg'.format(trainer.task.HYPER_PARAMS.src_img_dir), 
                          outf_nm=trainer.task.HYPER_PARAMS.src_tfrecord_path)
        self.dataset = create_tfdataset(tfrecord_file=trainer.task.HYPER_PARAMS.src_tfrecord_path+'.tfrecord', 
                                        shuffle_buffer=trainer.task.HYPER_PARAMS.shuffle_buffer, 
                                        epochs=self.epochs, 
                                        batch_size=self.batch_size, 
                                        pThreads=4)

        # create generator/discriminator nets
        self.g_net = Generator(dtype='float32')
        self.d_net = Discriminator(alpha=self.alpha, dtype='float32')

        # set up tensorboard writer
        self.tf_board_writer = tf.contrib.summary.create_file_writer(self.log_dir)
        self.tf_board_writer.set_as_default()

        # set up optimizers
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.g_learn_rate, 
                                                  beta1=self.beta1, beta2=self.beta2)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learn_rate,
                                                  beta1=self.beta1, beta2=self.beta2)

        # set up checkpoint directories
        self.global_step = tf.train.get_or_create_global_step()
        self.g_checkpoint_dir = os.path.join(self.model_dir, 'generator')
        self.g_root = tfe.Checkpoint(optimizer=self.g_optimizer, 
                                     model=self.g_net,
                                     optimizer_step=self.global_step)
        self.d_checkpoint_dir = os.path.join(self.model_dir, 'discriminator')
        self.d_root = tfe.Checkpoint(optimizer=self.d_optimizer, 
                                     model=self.d_net,
                                     optimizer_step=self.global_step)

        # restore generator/discriminator from previous checkpoints (if exist)
        if os.path.exists(self.model_dir):
            try:
                self.g_root.restore(tf.train.latest_checkpoint(self.g_checkpoint_dir))
            except Exception as ex:
                print('Could not load Generator model from {}'.format(self.g_checkpoint_dir))
            try:
                self.d_root.restore(tf.train.latest_checkpoint(self.d_checkpoint_dir))
            except Exception as ex:
                print('Could not load Discriminator model from {}'.format(self.d_checkpoint_dir))
            self.global_step = tf.train.get_or_create_global_step()
            print('Resuming training from latest checkpoint')
            print('Generator and Discriminator models loaded from global step: {}'.\
                  format(tf.train.get_or_create_global_step().numpy()))
        else:
            print('Model folder not found.')


    def train(self):

        for _, (batch_real_images) in enumerate(self.dataset):

            # construct random normal z input to feed into generator
            input_z = tf.random_normal(shape=(self.batch_size, self.z_size), dtype='float32')

            with tf.contrib.summary.record_summaries_every_n_global_steps(self.summary_per_n_steps):
                
                # define gradient tapes to start recording computation operations
                with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                    # run generator net with random normal z input to generate image batch
                    g_fake_images = self.g_net(input_z, is_training=True)
                    # run discriminator net with real input images
                    d_logits_real = self.d_net(batch_real_images, is_training=True)
                    # run discriminator net with generated fake images
                    d_logits_fake = self.d_net(g_fake_images, is_training=True)
                    # compute generator loss by feeding back the discriminator logits output
                    g_loss = self.g_net.compute_loss(d_logits_fake)
                    # compute discriminator hinge loss
                    d_loss = self.d_net.compute_loss(d_logits_real, d_logits_fake)

                # write losses to tensorboard as scalars & generated images
                tf.contrib.summary.scalar('generator_loss', g_loss)
                tf.contrib.summary.scalar('discriminator_loss', d_loss)
                tf.contrib.summary.image('generator_image', tf.to_float(g_fake_images), max_images=5)

                # get all discriminator variables (quantities to optimize)
                d_variables = self.d_net.variables
                # compute d(d_loss)/dx; where x is all discriminator variables
                d_grads = d_tape.gradient(d_loss, d_variables)

                # get all generator variables (quantities to optimize)
                g_variables = self.g_net.variables
                # compute d(g_loss)/dx; where x is all generator variables
                g_grads = g_tape.gradient(g_loss, g_variables)

                # update all variables
                self.d_optimizer.apply_gradients(zip(d_grads, d_variables),
                                                 global_step=self.global_step)
                self.g_optimizer.apply_gradients(zip(g_grads, g_variables),
                                                 global_step=self.global_step)

            # print training status
            counter = self.global_step.numpy()
            print('training step {}: discriminator loss {}; generator loss {}'.format(counter, d_loss, g_loss))

            # TRAINING PROCESS CONTROL FLOW
            # every 2000 steps, generate a batch of images
            if counter % 2000==0:
                print('Current step:{}'.format(counter))
                with tf.contrib.summary.always_record_summaries():
                    g_sample_images = self.g_net(input_z, is_training=False)
                    tf.contrib.summary.image('test_generator_image', 
                                             tf.to_float(g_sample_images), 
                                             max_images=16)
                print('Saving model snapshot')
                self.g_root.save(file_prefix=os.path.join(self.g_checkpoint_dir, "model.ckpt"))
                self.d_root.save(file_prefix=os.path.join(self.d_checkpoint_dir, "model.ckpt"))


            # save generator model at end of training
            if counter >= self.max_train_steps:
                self.g_root.save(file_prefix=os.path.join(self.g_checkpoint_dir, "model.ckpt"))
                self.d_root.save(file_prefix=os.path.join(self.d_checkpoint_dir, "model.ckpt"))


