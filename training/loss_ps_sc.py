#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_ps_sc.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Tue 16 Mar 2021 22:11:33 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
PS-SC Loss functions.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def G_logistic_ns(G, D, opt, training_set, minibatch_size, latent_type='uniform'):
    _ = opt
    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size, G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size, G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def calc_ps_loss(latents, delta_latents, reg1_out, reg2_out, C_delta_latents, C_lambda):
    reg12_avg = 0.5 * (reg1_out + reg2_out)
    var_mask = C_delta_latents > 0
    reg1_out_hat = tf.where(var_mask, reg1_out, reg12_avg)
    reg2_out_hat = tf.where(var_mask, reg2_out, reg12_avg)
    I_loss1 = tf.reduce_sum(tf.math.squared_difference(latents, reg1_out_hat), axis=1)
    I_loss2 = tf.reduce_sum(tf.math.squared_difference(delta_latents, reg2_out_hat), axis=1)
    I_loss = 0.5 * (I_loss1 + I_loss2)
    I_loss = autosummary('Loss/I_loss', I_loss)
    I_loss *= C_lambda
    return I_loss

def G_logistic_ns_ps_sc(G, D, I, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
                        C_lambda=1, epsilon=0.4, random_eps=False, use_cascade=False, cascade_dim=None):
    _ = opt
    C_global_size = G.input_shapes[0][1]

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    # Sample delta latents
    if use_cascade:
        C_delta_latents = tf.cast(tf.one_hot(cascade_dim, C_global_size), latents.dtype)
        C_delta_latents = tf.tile(C_delta_latents[tf.newaxis, :], [minibatch_size, 1])
        print('after onehot, C_delta_latents.shape:', C_delta_latents.get_shape().as_list())
    else:
        C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
        C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)

    if not random_eps:
        delta_target = C_delta_latents * epsilon
    else:
        epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
        delta_target = C_delta_latents * epsilon

    delta_latents = delta_target + latents

    labels = training_set.get_random_labels_tf(2*minibatch_size)
    latents_all = tf.concat([latents, delta_latents], axis=0)
    fake_all_out = G.get_output_for(latents_all, labels, is_training=True, return_atts=False)
    fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)

    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake1_out, labels, is_training=True)
    else:
        fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    regress_out = I.get_output_for(fake_all_out, is_training=True)
    reg1_out, reg2_out = tf.split(regress_out, 2, axis=0)
    I_loss = calc_ps_loss(latents, delta_latents, reg1_out, reg2_out, C_delta_latents, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss

    return G_loss, None

def calc_regress_loss(latents, pred_outs, C_global_size, C_lambda, minibatch_size, norm_ord=2):
    assert pred_outs.shape.as_list()[1] == (C_global_size)
    # Continuous latents loss
    G2_loss_C = tf.norm(pred_outs - latents, ord=norm_ord, axis=1)
    G2_loss = C_lambda * G2_loss_C
    return G2_loss

def G_logistic_ns_info_gan(G, D, I, opt, training_set, minibatch_size,
                           latent_type='uniform', C_lambda=1, norm_ord=2):
    _ = opt
    C_global_size = G.input_shapes[0][1]

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)

    fake_scores_out = D.get_output_for(fake_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    regress_out = I.get_output_for(fake_out, is_training=True)
    I_loss = calc_regress_loss(latents, regress_out, C_global_size, C_lambda,
                               minibatch_size, norm_ord=norm_ord)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss
    return G_loss, None

def D_logistic_r1_shared(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform'):
    _ = opt, training_set

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    fake_images_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg
