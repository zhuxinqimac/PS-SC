#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.loss_nav.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Tue 24 Aug 2021 02:06:15 AEST
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

def calc_minfeats_loss(feats, minfeats_lambda, mf_compare_idx):
    '''
    feats is a list of tensors of shapes:
    [2b, c, h, w] or [2b, c]
    '''
    loss = 0
    # print('len(feats):', len(feats))
    # print('mf_compare_idx:', mf_compare_idx)
    if mf_compare_idx is None:
        mf_compare_idx = list(range(len(feats)))
    assert max(mf_compare_idx) < len(feats)
    for i, feat in enumerate(feats):
        if i not in mf_compare_idx:
            continue
        feat_1, feat_2 = tf.split(feat, 2, axis=0)
        if len(feat_1.shape) > 2:
            feat_1 = tf.reduce_mean(feat_1, axis=[2,3]) # [b, nfeat]
            feat_2 = tf.reduce_mean(feat_2, axis=[2,3]) # [b, nfeat]
        loss_tmp = tf.reduce_mean(tf.math.squared_difference(feat_1, feat_2), axis=1)
        loss += loss_tmp
    return minfeats_lambda * loss

def nav_l2(N, G, I, opt, minibatch_size, C_lambda, if_train_I=False, epsilon=1, random_eps=True,
           dims_to_learn_ls=[0,1,2,3], minfeats_lambda=0, mf_compare_idx=[0,1,2], reg_lambda=1, eps_type='normal'):
    _ = opt
    z_latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]])
    # G.components.mapping.get_output_for(z_latents, None, is_training=False)
    dlatents = G.components.mapping.get_output_for(z_latents, None,
                                                   dlatent_broadcast=G.components.mapping.static_kwargs.dlatent_broadcast,
                                                   is_training=False)
    # dlatents: [b, w_dim]
    dirs = N.get_output_for(tf.reduce_mean(dlatents, axis=1)) # [n_lat, num_ws, w_dim] or [b, n_lat, num_ws, w_dim]

    # Sample delta
    n_lat = len(dims_to_learn_ls)
    delta_idx = tf.random.uniform([minibatch_size], minval=0, maxval=n_lat, dtype=tf.int32) # [b]
    dims_to_learn_tf = tf.constant(dims_to_learn_ls)
    if random_eps:
        if eps_type == 'normal':
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=1.0) # [b, 1]
        elif eps_type == 'signed':
            sign = (tf.cast(tf.random.uniform([minibatch_size, 1], minval=0, maxval=2, dtype=tf.int32), tf.float32) - 0.5) * 2 # [b, 1]
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=1.0) + 3 # [b, 1]
            epsilon = epsilon * sign
        else:
            raise ValueError('Unknown eps_type:', eps_type)

    if len(dirs.shape) == 4:
        # === Adaptive dir
        delta_idx_b = tf.stack([tf.range(minibatch_size), delta_idx], axis=1) # [b, 2]
        b_delta = epsilon[:, :, np.newaxis] * tf.gather_nd(dirs, delta_idx_b) # [b, num_ws, w_dim]
        print('using ada, b_delta.shape:', b_delta.shape)
    else:
        # === Static dir
        b_delta = epsilon[:, :, np.newaxis] * tf.gather(dirs, delta_idx, axis=0) # [b, num_ws, w_dim]
    dlatents_delta = b_delta + dlatents

    images_all = G.components.synthesis.get_output_for(tf.concat([dlatents, dlatents_delta], axis=0), is_training=False)
    # images, images_delta = tf.split(images_all, 2, axis=0)
    sh = images_all.shape.as_list()
    if sh[2] > I.input_shape[-1]:
        factor = sh[2] // I.input_shape[-1]
        images_all = tf.reduce_mean(tf.reshape(images_all, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

    outs = I.get_output_for(images_all, is_training=if_train_I, return_feats=(minfeats_lambda>0), return_as_list=True)
    regress_all = outs[0]
    reg_1, reg_2 = tf.split(regress_all, 2, axis=0)

    I_loss = calc_l2_loss(reg_1, reg_2, delta_idx, dims_to_learn_tf, epsilon, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    if reg_lambda > 0:
        reg_loss = tf.reduce_mean(tf.norm(tf.reshape(dirs, [n_lat, dirs.shape[1]*dirs.shape[2]]), axis=-1))
        reg_loss = autosummary('Loss/reg_loss', reg_loss)
        I_loss += reg_lambda * reg_loss

    if minfeats_lambda > 0:
        feats = outs[1:]
        F_loss = calc_minfeats_loss(feats, minfeats_lambda, mf_compare_idx)
        F_loss = autosummary('Loss/F_loss', F_loss)
        I_loss += F_loss

    return I_loss
