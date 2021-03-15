#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: group_verify.py
# --- Creation Date: 25-08-2020
# --- Last Modified: Thu 27 Aug 2020 22:47:47 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Verify group representations.
"""

import os
import math
import numpy as np
import pdb
import time
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base
from metrics.perceptual_path_length import normalize, slerp
from training import misc
from training.utils import get_return_v


class GVerify(metric_base.MetricBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, I_net, Gs, **kwargs):
        E = I_net.clone(is_validation=True)
        G = Gs.clone(is_validation=True)
        n_samples = 64

        # Element e verify.
        fake_e, gfeat_G_e = get_return_v(G.run(np.zeros([1] + G.input_shape[1:]),
                                               np.zeros([1, 0]), is_validation=True), 2)
        mean_e, var_e, gfeat_E_e = get_return_v(E.run(fake_e, np.zeros([1, 0]),
                                                      is_validation=True), 3)
        gfeat_dim = int(math.sqrt(gfeat_G_e.shape[1]))
        assert gfeat_dim * gfeat_dim == gfeat_G_e.shape[1]
        gfeat_G_e = np.reshape(gfeat_G_e, [1, gfeat_dim, gfeat_dim])
        gfeat_E_e = np.reshape(gfeat_E_e, [1, gfeat_dim, gfeat_dim])
        gfeat_eye = np.eye(gfeat_dim)[np.newaxis, :]
        print('gfeat_G_e:', gfeat_G_e)
        print('gfeat_E_e:', gfeat_E_e)
        print('mean_e:', mean_e)
        print('var_e:', var_e)
        dist_eye_G = np.sum(np.square(gfeat_G_e - gfeat_eye))
        dist_eye_E = np.sum(np.square(gfeat_E_e - gfeat_eye))
        dist_eye_rec = np.sum(np.square(gfeat_E_e - gfeat_G_e))
        self._report_result(dist_eye_G, suffix='_dist_eye_G')
        self._report_result(dist_eye_E, suffix='_dist_eye_E')
        self._report_result(dist_eye_rec, suffix='_dist_eye_rec')

        # Inverse verify.
        lats_p = np.random.normal(size=[n_samples] + G.input_shape[1:])
        lats_n = -lats_p
        lats_pn = np.concatenate((lats_p, lats_n), axis=0)
        fake_pn, gfeat_G_pn = get_return_v(G.run(lats_pn,
                                                 np.zeros([1, 0]), is_validation=True), 2)
        mean_pn, var_pn, gfeat_E_pn = get_return_v(E.run(fake_pn, np.zeros([1, 0]),
                                                         is_validation=True), 3)
        gfeat_G_pn = np.reshape(gfeat_G_pn, [2*n_samples, gfeat_dim, gfeat_dim])
        gfeat_E_pn = np.reshape(gfeat_E_pn, [2*n_samples, gfeat_dim, gfeat_dim])
        gfeat_G_mul = np.matmul(gfeat_G_pn[:n_samples], gfeat_G_pn[n_samples:])
        gfeat_E_mul = np.matmul(gfeat_E_pn[:n_samples], gfeat_E_pn[n_samples:])
        print('gfeat_G_p', gfeat_G_pn[0])
        print('gfeat_G_n', gfeat_G_pn[n_samples])
        print('gfeat_G_mul', gfeat_G_mul[0])
        print('gfeat_E_p', gfeat_E_pn[0])
        print('gfeat_E_n', gfeat_E_pn[n_samples])
        print('gfeat_E_mul', gfeat_E_mul[0])
        dist_eye_G_mul = np.mean(np.sum(np.square(gfeat_G_mul - gfeat_eye), axis=(1,2)))
        dist_eye_E_mul = np.mean(np.sum(np.square(gfeat_E_mul - gfeat_eye), axis=(1,2)))
        dist_G_pn = np.mean(np.sum(np.square(gfeat_G_pn[:n_samples] - gfeat_G_pn[n_samples:]), axis=(1,2)))
        self._report_result(dist_eye_G_mul, suffix='_dist_eye_G_mul')
        self._report_result(dist_eye_E_mul, suffix='_dist_eye_E_mul')
        self._report_result(dist_G_pn, suffix='_dist_G_pn')

        # Multiplication verify.
        lats_1 = np.random.normal(size=[n_samples] + G.input_shape[1:])
        lats_2 = np.random.normal(size=[n_samples] + G.input_shape[1:])
        lats_3 = lats_1 + lats_2
        lats_123 = np.concatenate((lats_1, lats_2, lats_3), axis=0)
        fake_123, gfeat_G_123 = get_return_v(G.run(lats_123,
                                                 np.zeros([1, 0]), is_validation=True), 2)
        mean_123, var_123, gfeat_E_123 = get_return_v(E.run(fake_123, np.zeros([1, 0]),
                                                         is_validation=True), 3)
        gfeat_G_123 = np.reshape(gfeat_G_123, [3*n_samples, gfeat_dim, gfeat_dim])
        gfeat_E_123 = np.reshape(gfeat_E_123, [3*n_samples, gfeat_dim, gfeat_dim])
        gfeat_G_mul_12 = np.matmul(gfeat_G_123[:n_samples], gfeat_G_123[n_samples:2*n_samples])
        gfeat_E_mul_12 = np.matmul(gfeat_E_123[:n_samples], gfeat_E_123[n_samples:2*n_samples])
        gfeat_G_3 = gfeat_G_123[2*n_samples:]
        gfeat_E_3 = gfeat_E_123[2*n_samples:]
        print('gfeat_G_mul_12', gfeat_G_mul_12[0])
        print('gfeat_G_3', gfeat_G_3[0])
        print('gfeat_E_mul_12', gfeat_E_mul_12[0])
        print('gfeat_E_3', gfeat_E_3[0])
        dist_G_mul = np.mean(np.sum(np.square(gfeat_G_mul_12 - gfeat_G_3), axis=(1,2)))
        dist_E_mul = np.mean(np.sum(np.square(gfeat_E_mul_12 - gfeat_E_3), axis=(1,2)))
        self._report_result(dist_G_mul, suffix='_dist_G_mul')
        self._report_result(dist_E_mul, suffix='_dist_E_mul')

        return {'dist_eye_G': dist_eye_G, 'dist_eye_E': dist_eye_E,
                'dist_eye_rec': dist_eye_rec,
                'dist_eye_G_mul': dist_eye_G_mul,
                'dist_eye_E_mul': dist_eye_E_mul,
                'dist_G_pn': dist_G_pn,
                'dist_G_mul': dist_G_mul,
                'dist_E_mul': dist_E_mul,
                }
