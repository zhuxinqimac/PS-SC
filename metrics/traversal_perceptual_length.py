#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: traversal_perceptual_length.py
# --- Creation Date: 12-05-2020
# --- Last Modified: Mon 15 Feb 2021 17:10:10 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Traversal Perceptual Length (TPL)."""

import os
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

#----------------------------------------------------------------------------

class TPL(metric_base.MetricBase):
    def __init__(self, n_samples_per_dim, crop, Gs_overrides, n_traversals, no_mapping, no_convert=False, active_thresh=0.1, use_bound_4=True, **kwargs):
        super().__init__(**kwargs)
        self.crop = crop
        self.Gs_overrides = Gs_overrides
        self.n_samples_per_dim = n_samples_per_dim
        self.n_traversals = n_traversals
        self.no_mapping = no_mapping
        self.no_convert = no_convert
        self.active_thresh = active_thresh
        self.use_bound_4 = use_bound_4

    def _evaluate(self, Gs, Gs_kwargs, num_gpus, **kwargs):
        Gs_kwargs = dict(Gs_kwargs)
        Gs_kwargs.update(self.Gs_overrides)
        minibatch_per_gpu = (self.n_samples_per_dim - 1) // num_gpus + 1
        if (not self.no_mapping) and (not self.no_convert):
            Gs = Gs.convert(new_func_name='training.vc_networks2.G_main_vc2')

        # Construct TensorFlow graph.
        n_continuous = Gs.input_shape[1]
        distance_expr = []
        eval_dim_phs = []
        lat_start_alpha_phs = []
        lat_end_alpha_phs = []
        lat_sample_phs = []
        lerps_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                if self.no_mapping:
                    noise_vars = [var for name, var in Gs_clone.vars.items() if name.startswith('noise')]
                else:
                    noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if name.startswith('noise')]

                # Latent pairs placeholder
                eval_dim = tf.placeholder(tf.int32)
                lat_start_alpha = tf.placeholder(tf.float32) # should be in [0, 1]
                lat_end_alpha = tf.placeholder(tf.float32) # should be in [0, 1]
                eval_dim_phs.append(eval_dim)
                lat_start_alpha_phs.append(lat_start_alpha)
                lat_end_alpha_phs.append(lat_end_alpha)
                eval_dim_mask = tf.tile(tf.one_hot(eval_dim, n_continuous)[tf.newaxis, :] > 0, [minibatch_per_gpu, 1])
                lerp_t = tf.linspace(lat_start_alpha, lat_end_alpha, minibatch_per_gpu) # [b]
                lerps_expr.append(lerp_t)
                lat_sample = tf.placeholder(tf.float32, shape=Gs_clone.input_shape[1:])
                lat_sample_phs.append(lat_sample)

                # lat_t0 = tf.zeros([minibatch_per_gpu] + Gs_clone.input_shape[1:])
                lat_t0 = tf.tile(lat_sample[tf.newaxis, :], [minibatch_per_gpu, 1])
                if self.use_bound_4:
                    lat_t0_min2 = tf.zeros_like(lat_t0) - 4
                else:
                    lat_t0_min2 = lat_t0 - 2
                lat_t0 = tf.where(eval_dim_mask, lat_t0_min2, lat_t0) # [b, n_continuous]

                lat_t1 = tf.tile(lat_sample[tf.newaxis, :], [minibatch_per_gpu, 1])
                if self.use_bound_4:
                    lat_t1_add2 = tf.zeros_like(lat_t1) + 4
                else:
                    lat_t1_add2 = lat_t1 + 2
                lat_t1 = tf.where(eval_dim_mask, lat_t1_add2, lat_t1) # [b, n_continuous]
                lat_e = tflib.lerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis]) # [b, n_continuous]

                # labels = tf.reshape(self._get_random_labels_tf(minibatch_per_gpu), [minibatch_per_gpu, -1])
                labels = tf.zeros([minibatch_per_gpu, 0], dtype=tf.float32)
                if self.no_mapping:
                    dlat_e = lat_e
                else:
                    dlat_e = get_return_v(Gs_clone.components.mapping.get_output_for(lat_e, labels, **Gs_kwargs), 1)

                # Synthesize images.
                with tf.control_dependencies([var.initializer for var in noise_vars]): # use same noise inputs for the entire minibatch
                    if self.no_mapping:
                        images = get_return_v(Gs_clone.get_output_for(dlat_e, labels, randomize_noise=False, **Gs_kwargs), 1)
                    else:
                        images = get_return_v(Gs_clone.components.synthesis.get_output_for(dlat_e, randomize_noise=False, **Gs_kwargs), 1)
                    # print('images.shape:', images.get_shape().as_list())
                    images = tf.cast(images, tf.float32)

                # Crop only the face region.
                if self.crop:
                    c = int(images.shape[2] // 8)
                    images = images[:, :, c*3 : c*7, c*2 : c*6]

                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                factor = images.shape[2] // 256
                if factor > 1:
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3,5])

                # Scale dynamic range from [-1,1] to [0,255] for VGG.
                images = (images + 1) * (255 / 2)

                # Evaluate perceptual distance.
                if images.get_shape().as_list()[1] == 1:
                    images = tf.tile(images, [1, 3, 1, 1])
                img_e0 = images[:-1]
                img_e1 = images[1:]
                distance_measure = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')
                distance_tmp = distance_measure.get_output_for(img_e0, img_e1)
                print('distance_tmp.shape:', distance_tmp.get_shape().as_list())
                distance_expr.append(distance_tmp)

        # Sampling loop
        n_segs_per_dim = (self.n_samples_per_dim - 1) // ((minibatch_per_gpu - 1) * num_gpus)
        self.n_samples_per_dim = n_segs_per_dim * ((minibatch_per_gpu - 1) * num_gpus) + 1
        alphas = np.linspace(0., 1., num=(n_segs_per_dim * num_gpus)+1)
        traversals_dim = []
        for n in range(self.n_traversals):
            lat_sample_np = np.random.normal(size=Gs_clone.input_shape[1:])
            all_distances = []
            sum_distances = []
            for i in range(n_continuous):
                self._report_progress(i, n_continuous)
                dim_distances = []
                for j in range(n_segs_per_dim):
                    fd = {}
                    for k_gpu in range(num_gpus):
                        fd.update({eval_dim_phs[k_gpu]:i,
                                   lat_start_alpha_phs[k_gpu]:alphas[j*num_gpus+k_gpu],
                                   lat_end_alpha_phs[k_gpu]:alphas[j*num_gpus+k_gpu+1],
                                   lat_sample_phs[k_gpu]:lat_sample_np})
                    distance_expr_out, lerps_expr_out = tflib.run([distance_expr, lerps_expr], feed_dict=fd)
                    dim_distances += distance_expr_out
                    # dim_distances += tflib.run(distance_expr, feed_dict=fd)
                    # print(lerps_expr_out)
                dim_distances = np.concatenate(dim_distances, axis=0)
                # print('dim_distances.shape:', dim_distances.shape)
                all_distances.append(dim_distances)
                sum_distances.append(np.sum(dim_distances))
            traversals_dim.append(sum_distances)
        traversals_dim = np.array(traversals_dim) # shape: (n_traversals, n_continuous)
        avg_distance_per_dim = np.mean(traversals_dim, axis=0)
        std_distance_per_dim = np.std(traversals_dim, axis=0)
        # pdb.set_trace()
        active_mask = np.array(avg_distance_per_dim) > self.active_thresh
        active_distances = np.extract(active_mask, avg_distance_per_dim)
        active_stds = np.extract(active_mask, std_distance_per_dim)
        sum_distance = np.sum(active_distances)
        mean_distance = np.sum(active_distances) / len(avg_distance_per_dim)
        mean_std = np.sum(active_stds) / len(avg_distance_per_dim)
        norm_dis_std = np.sqrt(mean_distance*mean_distance + mean_std*mean_std)
        print('avg distance per dim:', avg_distance_per_dim)
        print('std distance per dim:', std_distance_per_dim)
        print('sum_distance:', sum_distance)
        print('mean_distance:', mean_distance)
        print('mean_std:', mean_std)
        print('norm_dis_std:', norm_dis_std)
        # def _report_result(self, value, suffix='', fmt='%-10.4f'):
        self._report_result(sum_distance, suffix='_sum_dist')
        self._report_result(mean_distance, suffix='_mean_dist')
        self._report_result(mean_std, suffix='_mean_std')
        self._report_result(norm_dis_std, suffix='_norm_dist_std')
        # pdb.set_trace()
        return {'tpl_per_dim': avg_distance_per_dim}
#----------------------------------------------------------------------------
