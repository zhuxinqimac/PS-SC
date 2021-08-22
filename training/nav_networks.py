#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.nav_networks.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Sun 22 Aug 2021 16:29:04 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator networks.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from training.networks_stylegan2 import dense_layer
from training.networks_stylegan2 import apply_bias_act
from dnnlib import EasyDict
from training.networks_stylegan2 import get_weight

def get_n_att(nav_type, num_ws):
    token_ls = nav_type.split('@')
    if len(token_ls) >= 2:
        n_att_ws = int(token_ls[1])
        assert n_att_ws <= num_ws
    else:
        n_att_ws = num_ws
    return n_att_ws

def navigator(w, nav_type='linear', n_lat=20, num_ws=18, w_dim=512,
              **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    w: [b, w_dim]
    '''
    w.set_shape([None, w_dim])
    w = tf.cast(w, 'float32')
    b = tf.shape(w)[0]

    with tf.variable_scope('Nav_var'):
        if nav_type == 'linear':
            # dirs = get_weight([n_lat, num_ws * w_dim], use_wscale=False, weight_var='w_directions')
            init = tf.initializers.zeros()
            dirs = tf.get_variable('w_directions', shape=[n_lat, num_ws*w_dim], initializer=init)
        elif nav_type.startswith('layersoft'):
            n_att_ws = get_n_att(nav_type, num_ws)
            init = tf.initializers.random_normal(0, 1)
            layer_att = tf.get_variable('layer_att', shape=[n_lat, n_att_ws], initializer=init)
            layer_softmax = tf.concat([tf.nn.softmax(layer_att, axis=-1), tf.zeros([n_lat, num_ws - n_att_ws], dtype='float32')], axis=-1)
            per_layer_dir = tf.get_variable('per_layer_dir', shape=[n_lat, w_dim], initializer=init)
            dirs = layer_softmax[:, :, np.newaxis] * per_layer_dir[:, np.newaxis, ...]
        elif nav_type.startswith('layersoftelasticunit'):
            n_att_ws = get_n_att(nav_type, num_ws)
            init = tf.initializers.random_normal(0, 1)
            layer_att = tf.get_variable('layer_att', shape=[n_lat, n_att_ws], initializer=init)
            layer_softmax = tf.concat([tf.nn.softmax(layer_att, axis=-1), tf.zeros([n_lat, num_ws - n_att_ws], dtype='float32')], axis=-1)
            per_layer_dir = tf.get_variable('per_layer_dir', shape=[n_lat, w_dim], initializer=init)
            per_layer_dir_unit, _ = tf.linalg.normalize(per_layer_dir, axis=-1)
            len_dir = tf.get_variable('len_dir', shape=[n_lat], initializer=init)**2
            dirs = layer_softmax[:, :, np.newaxis] * per_layer_dir_unit[:, np.newaxis, ...] * len_dir[:, np.newaxis, np.newaxis]
        elif nav_type.startswith('adalayersoft'):
            # Directions depends on the input w.
            n_att_ws = get_n_att(nav_type, num_ws)
            with tf.variable_scope('layer_att'):
                layer_att = apply_bias_act(dense_layer(w, fmaps=n_lat * n_att_ws), act='linear') # [b, n_lat * n_att_ws]
            layer_att = tf.reshape(layer_att, [b, n_lat, n_att_ws])
            layer_softmax = tf.concat([tf.nn.softmax(layer_att, axis=-1),
                                       tf.zeros([b, n_lat, num_ws - n_att_ws], dtype='float32')], axis=-1) # [b, n_lat, num_ws]
            with tf.variable_scope('per_layer_dir'):
                per_layer_dir = apply_bias_act(dense_layer(w, fmaps=n_lat * w_dim), act='linear') # [b, n_lat * w_dim]
            per_layer_dir = tf.reshape(per_layer_dir, [b, n_lat, w_dim])
            dirs = layer_softmax[:, :, :, np.newaxis] * per_layer_dir[:, :, np.newaxis, ...] # [b, n_lat, num_ws, w_dim]
        else:
            raise ValueError('Unknown nav_type:', nav_type)

    if nav_type.startswith('ada'):
        dirs = tf.reshape(dirs, [b, n_lat, num_ws, w_dim])
    else:
        dirs = tf.reshape(dirs, [n_lat, num_ws, w_dim])
    return dirs
