#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.loss_nav.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Sat 14 Aug 2021 22:01:57 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigation loss.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def calc_l2_loss(reg_1, reg_2, delta_idx, dims_to_learn_tf, epsilon, l2_lambda):
    '''
    delta_idx: [b]
    '''
    reg_delta = reg_2 - reg_1
    I_delta_idx = tf.gather(dims_to_learn_tf, delta_idx, axis=0) # [b]
    target_delta = epsilon * tf.cast(tf.one_hot(I_delta_idx, reg_delta.shape[-1]), reg_delta.dtype)
    # target_delta = tf.cast(tf.one_hot(I_delta_idx, reg_delta.shape[-1]), reg_delta.dtype)
    I_loss = l2_lambda * tf.reduce_sum(tf.math.squared_difference(target_delta, reg_delta), axis=1)
    return I_loss

def nav_l2(N, G, I, opt, minibatch_size, C_lambda, if_train_I=False, epsilon=1, random_eps=True, dims_to_learn_ls=[0,1,2,3], reg_lambda=1):
    _ = opt
    z_latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]])
    # G.components.mapping.get_output_for(z_latents, None, is_training=False)
    dlatents = G.components.mapping.get_output_for(z_latents, None,
                                                   dlatent_broadcast=G.components.mapping.static_kwargs.dlatent_broadcast,
                                                   is_training=False)
    # dlatents: [b, w_dim]
    dirs = N.get_output_for(tf.reduce_mean(dlatents, axis=1)) # [n_lat, num_ws, w_dim]

    # Sample delta
    n_lat = len(dims_to_learn_ls)
    delta_idx = tf.random.uniform([minibatch_size], minval=0, maxval=n_lat, dtype=tf.int32) # [b]
    dims_to_learn_tf = tf.constant(dims_to_learn_ls)
    # C_delta_latents = tf.cast(tf.one_hot(delta_idx, n_lat), dlatents.dtype) # [b, n_lat]

    # if not random_eps:
        # delta_target = C_delta_latents * epsilon
    # else:
        # epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=1.0) # [b, 1]
        # delta_target = C_delta_latents * epsilon
    if random_eps:
        epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=1.0) # [b, 1]

    b_delta = epsilon[:, :, np.newaxis] * tf.gather(dirs, delta_idx, axis=0) # [b, num_ws, w_dim]
    dlatents_delta = b_delta + dlatents

    images_all = G.components.synthesis.get_output_for(tf.concat([dlatents, dlatents_delta], axis=0), is_training=False)
    # images, images_delta = tf.split(images_all, 2, axis=0)
    sh = images_all.shape.as_list()
    if sh[2] > I.input_shape[-1]:
        factor = sh[2] // I.input_shape[-1]
        images_all = tf.reduce_mean(tf.reshape(images_all, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
    regress_all = I.get_output_for(images_all, is_training=if_train_I)
    reg_1, reg_2 = tf.split(regress_all, 2, axis=0)

    I_loss = calc_l2_loss(reg_1, reg_2, delta_idx, dims_to_learn_tf, epsilon, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    reg_loss = tf.reduce_mean(tf.norm(tf.reshape(dirs, [n_lat, dirs.shape[1]*dirs.shape[2]]), axis=-1))
    reg_loss = autosummary('Loss/reg_loss', reg_loss)
    I_loss += reg_lambda * reg_loss

    return I_loss
