#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.nav_networks.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Sat 14 Aug 2021 04:17:42 AEST
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
    dirs = tf.reshape(dirs, [n_lat, num_ws, w_dim])
    return dirs
