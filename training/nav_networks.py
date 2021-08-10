#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training.nav_networks.py
# --- Creation Date: 10-08-2021
# --- Last Modified: Tue 10 Aug 2021 16:59:13 AEST
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

def navigator(w, nav_type, n_lat, num_ws, w_dim):
    '''
    w: [b, w_dim]
    '''
    w.set_shape([None, w_dim])
    w = tf.cast(w, 'float32')

    if nav_type == 'linear':
        dirs = get_weight([n_lat, num_ws * w_dim], weight_var='w_directions')
    dirs = tf.reshape(dirs, [n_lat, num_ws, w_dim])
    return dirs
