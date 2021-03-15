#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: loss_vc2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Mon 15 Mar 2021 17:19:39 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Loss function in VC2.
"""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def G_logistic_ns(G, D, opt, training_set, minibatch_size, DM=None, latent_type='uniform'):
    _ = opt
    # latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size, G.input_shapes[0][1]], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size, G.input_shapes[0][1]])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size, G.input_shapes[0][1]])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type):
    assert regress_out.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    if delta_type == 'onedim':
        prob_C = tf.nn.softmax(regress_out[:, D_global_size:], axis=1)
        I_loss_C = C_delta_latents * tf.log(prob_C + 1e-12)
        I_loss_C = C_lambda * I_loss_C

        I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
        I_loss = - I_loss_C

        # Continuous latents loss
        # I_loss_C = tf.nn.softmax_cross_entropy_with_logits_v2(C_delta_latents,
                                                              # regress_out, axis=1, name='delta_regress_loss')
        # I_loss = C_lambda * I_loss_C
    elif delta_type == 'fulldim':
        I_loss_C = tf.reduce_sum((tf.nn.sigmoid(regress_out[:, D_global_size:]) - C_delta_latents) ** 2, axis=1)
        I_loss = C_lambda * I_loss_C
    return I_loss

def G_logistic_ns_vc2(G, D, I, opt, training_set, minibatch_size, DM, I_info=None, latent_type='uniform',
                     D_global_size=0, D_lambda=0, C_lambda=1, epsilon=0.4,
                     random_eps=False, delta_type='onedim', own_I=False):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    # Sample delta latents
    if delta_type == 'onedim':
        C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
        C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)
    elif delta_type == 'fulldim':
        C_delta_latents = tf.random.uniform([minibatch_size, C_global_size], minval=0, maxval=1.0, dtype=latents.dtype)

    if delta_type == 'onedim':
        if not random_eps:
            delta_target = C_delta_latents * epsilon
        else:
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
            delta_target = C_delta_latents * epsilon
    else:
        delta_target = (C_delta_latents - 0.5) * epsilon

    delta_latents = delta_target + latents

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
        delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_latents], axis=1)

    # labels = training_set.get_random_labels_tf(minibatch_size)

    # if own_I:
        # fake1_out, atts = G.get_output_for(latents, labels, is_training=True, return_atts=True)
    # else:
        # fake1_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    # fake2_out = G.get_output_for(delta_latents, labels, is_training=True, return_atts=False)

    labels = training_set.get_random_labels_tf(2*minibatch_size)
    latents_all = tf.concat([latents, delta_latents], axis=0)
    if own_I:
        fake_all_out, atts_all = G.get_output_for(latents_all, labels, is_training=True, return_atts=True)
        fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)
        atts = atts_all[:minibatch_size]
    else:
        fake_all_out = G.get_output_for(latents_all, labels, is_training=True)
        fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)

    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake1_out, labels, is_training=True)
    else:
        fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    if own_I:
        regress_out = I.get_output_for(fake1_out, fake2_out, atts, is_training=True)
        # regress_out = regress_out[:, ::-1]
    else:
        regress_out = I.get_output_for(fake1_out, fake2_out, is_training=True)
    I_loss = calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type)
    # I_loss = calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss

    return G_loss, None

def calc_vc_byvae_loss(latents, delta_latents, reg1_out, reg2_out, C_delta_latents,
                       D_global_size, C_global_size, D_lambda, C_lambda, delta_type):
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

def G_logistic_byvae_ns_vc2(G, D, I, opt, training_set, minibatch_size, DM=None, I_info=None, latent_type='uniform',
                     D_global_size=0, D_lambda=0, C_lambda=1, epsilon=0.4,
                     random_eps=False, delta_type='onedim', own_I=False,
                     use_cascade=False, cascade_dim=None):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    # Sample delta latents
    if delta_type == 'onedim':
        if use_cascade:
            C_delta_latents = tf.cast(tf.one_hot(cascade_dim, C_global_size), latents.dtype)
            C_delta_latents = tf.tile(C_delta_latents[tf.newaxis, :], [minibatch_size, 1])
            print('after onehot, C_delta_latents.shape:', C_delta_latents.get_shape().as_list())
        else:
            C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
            C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)
    elif delta_type == 'fulldim':
        C_delta_latents = tf.random.uniform([minibatch_size, C_global_size], minval=0, maxval=1.0, dtype=latents.dtype)

    if delta_type == 'onedim':
        if not random_eps:
            delta_target = C_delta_latents * epsilon
        else:
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
            delta_target = C_delta_latents * epsilon
    else:
        delta_target = (C_delta_latents - 0.5) * epsilon

    delta_latents = delta_target + latents

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
        delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_latents], axis=1)

    # labels = training_set.get_random_labels_tf(minibatch_size)

    # if own_I:
        # fake1_out, atts = G.get_output_for(latents, labels, is_training=True, return_atts=True)
    # else:
        # fake1_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)
    # fake2_out = G.get_output_for(delta_latents, labels, is_training=True, return_atts=False)

    labels = training_set.get_random_labels_tf(2*minibatch_size)
    latents_all = tf.concat([latents, delta_latents], axis=0)
    if own_I:
        fake_all_out, atts_all = G.get_output_for(latents_all, labels, is_training=True, return_atts=True)
        atts = atts_all[:minibatch_size]
    else:
        fake_all_out = G.get_output_for(latents_all, labels, is_training=True, return_atts=False)
    fake1_out, fake2_out = tf.split(fake_all_out, 2, axis=0)

    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake1_out, labels, is_training=True)
    else:
        fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    if own_I:
        regress_out = I.get_output_for(fake_all_out, atts_all, is_training=True)
        # regress_out = regress_out[:, ::-1]
    else:
        regress_out = I.get_output_for(fake_all_out, is_training=True)
    reg1_out, reg2_out = tf.split(regress_out, 2, axis=0)
    I_loss = calc_vc_byvae_loss(latents, delta_latents, reg1_out, reg2_out, C_delta_latents,
                                D_global_size, C_global_size, D_lambda, C_lambda, delta_type)
    # I_loss = calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss

    return G_loss, None

def calc_regress_loss(clatents, pred_outs, D_global_size, C_global_size, D_lambda, C_lambda,
                      minibatch_size, norm_ord=2, n_dim_strict=0, loose_rate=0.2):
    assert pred_outs.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    # G2_loss_C = tf.reduce_sum((pred_outs[:] - clatents) ** 2, axis=1)

    # Only n_dim_strict == full or 1 are supported now.
    if n_dim_strict == 1:
        # print('using n_dim_strict==1')
        dropped_dim = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
        dropped_dim = tf.cast(tf.one_hot(dropped_dim, C_global_size), pred_outs.dtype)
        # pred_outs = pred_outs * (1 - dropped_dim)
        # clatents = clatents * (1 - clatents)
    else:
        dropped_dim = tf.ones([minibatch_size, C_global_size], dtype=pred_outs.dtype)
    # G2_loss_C = tf.norm(pred_outs - clatents, ord=norm_ord, axis=1)
    G2_loss_C = tf.norm(dropped_dim * (pred_outs - clatents) + loose_rate * (1 - dropped_dim) * (pred_outs - clatents),
                        ord=norm_ord, axis=1)
    G2_loss = C_lambda * G2_loss_C
    return G2_loss

def calc_regress_grow_loss(clatents, pred_outs, D_global_size, C_global_size, D_lambda, C_lambda, opt_reset_ls):
    assert pred_outs.shape.as_list()[1] == (D_global_size + C_global_size)
    print('opt_reset_ls:', opt_reset_ls)
    opt_reset_tf = tf.constant(opt_reset_ls[::-1], dtype=tf.float32)
    opt_reset_tf_mask = tf.reshape(opt_reset_tf, [1, len(opt_reset_ls), 1])
    opt_reset_tf_mask = tf.tile(opt_reset_tf_mask, [1, 1, C_global_size // len(opt_reset_ls)])
    opt_reset_tf_mask = tf.reshape(opt_reset_tf_mask, [1, C_global_size])
    g_step = tf.train.get_global_step()
    opt_reset_tf_mask = opt_reset_tf_mask <= tf.cast(g_step, tf.float32)
    opt_reset_tf_mask = tf.cast(opt_reset_tf_mask, dtype=clatents.dtype)
    # Continuous latents loss
    # squared = ((pred_outs - clatents) ** 2) * opt_reset_tf_mask
    squared = ((pred_outs - clatents) ** 2) * 0
    G2_loss_C = tf.reduce_sum(squared, axis=1)
    G2_loss = C_lambda * G2_loss_C
    return G2_loss

def calc_outlier_loss(outlier, pred_outs, D_global_size, C_global_size, D_lambda, C_lambda):
    assert pred_outs.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    G2_loss_C = tf.nn.softmax_cross_entropy_with_logits_v2(outlier, pred_outs, axis=1, name='outlier_loss')
    G2_loss = C_lambda * G2_loss_C
    return G2_loss

def calc_regress_and_att_loss(clatents, pred_outs, atts, gen_atts, D_global_size, C_global_size,
                              D_lambda, C_lambda, att_lambda):
    assert pred_outs.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    G2_loss_C_pred = tf.reduce_sum((pred_outs - clatents) ** 2, axis=1)
    G2_loss_pred = C_lambda * G2_loss_C_pred
    G2_loss_pred = autosummary('Loss/G2_loss_pred', G2_loss_pred)

    # Continuous gen_atts loss
    G2_loss_C_atts = tf.reduce_mean((gen_atts - atts) ** 2, axis=[2,3,4])
    G2_loss_C_atts = tf.reduce_sum(G2_loss_C_atts, axis=1)
    G2_loss_atts = att_lambda * G2_loss_C_atts
    G2_loss_atts = autosummary('Loss/G2_loss_atts', G2_loss_atts)

    G2_loss = G2_loss_pred + G2_loss_atts

    return G2_loss

def G_logistic_ns_vc2_info_gan2(G, D, I, opt, training_set, minibatch_size, DM=None,
                               latent_type='uniform', D_global_size=0, D_lambda=0,
                               C_lambda=1, norm_ord=2, n_dim_strict=0, loose_rate=0.2):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        clatents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        clatents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        clatents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, clatents], axis=1)
    else:
        latents = clatents

    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_out = G.get_output_for(latents, labels, is_training=True, return_atts=False)

    fake_scores_out = D.get_output_for(fake_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    regress_out = I.get_output_for(fake_out, is_training=True)
    # I_loss = calc_regress_grow_loss(clatents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, opt_reset_ls)
    I_loss = calc_regress_loss(clatents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda,
                               minibatch_size, norm_ord=norm_ord, n_dim_strict=n_dim_strict, loose_rate=loose_rate)
    I_loss = autosummary('Loss/I_loss', I_loss)

    G_loss += I_loss
    return G_loss, None

def D_logistic_r1_vc2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

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

def D_logistic_r1_vc2_info_gan2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

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
