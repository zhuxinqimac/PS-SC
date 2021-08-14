#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.nav_networks.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Sat 14 Aug 2021 15:29:25 AEST
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
from dnnlib import EasyDict
from training.networks_stylegan2 import get_weight

def navigator(w, nav_type='linear', n_lat=20, num_ws=18, w_dim=512,
              **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    '''
    w: [b, w_dim]
    '''
    w.set_shape([None, w_dim])
    w = tf.cast(w, 'float32')

    with tf.variable_scope('Nav_var'):
        if nav_type == 'linear':
            # dirs = get_weight([n_lat, num_ws * w_dim], use_wscale=False, weight_var='w_directions')
            init = tf.initializers.zeros()
            dirs = tf.get_variable('w_directions', shape=[n_lat, num_ws*w_dim], initializer=init)
        elif nav_type == 'layersoft':
            init = tf.initializers.random_normal(0, 1)
            layer_att = tf.get_variable('layer_att', shape=[n_lat, num_ws], initializer=init)
            layer_softmax = tf.nn.softmax(layer_att, axis=-1)
            per_layer_dir = tf.get_variable('per_layer_dir', shape=[n_lat, w_dim], initializer=init)
            dirs = layer_softmax[:, :, np.newaxis] * per_layer_dir[:, np.newaxis, ...]
        else:
            raise ValueError('Unknown nav_type:', nav_type)
    dirs = tf.reshape(dirs, [n_lat, num_ws, w_dim])
    return dirs
