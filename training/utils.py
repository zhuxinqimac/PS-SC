#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 14-08-2020
# --- Last Modified: Sat 07 Aug 2021 16:57:01 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Useful functions.
"""
import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from training import misc
from scipy.stats import truncnorm

def get_return_v(x, topk=1):
    if (not isinstance(x, tuple)) and (not isinstance(x, list)):
        return x if topk == 1 else tuple([x] + [None] * (topk - 1))
    if topk > len(x):
        return tuple(list(x) + [None] * (topk - len(x)))
    else:
        if topk == 1:
            return x[0]
        else:
            return tuple(x[:topk])
    # if topk == 1:
        # return x[0]
    # else:
        # return x[:topk]

def append_gfeats(latents, G):
    len_gfeats = sum(G.static_kwargs.subgroup_sizes_ls)
    b = latents.shape[0]
    new_latents = np.concatenate((latents, np.zeros([b, len_gfeats], dtype=latents.dtype)), axis=1)
    return new_latents

def save_atts(atts, filename, grid_size, drange, grid_fakes, n_samples_per):
    canvas = np.zeros([grid_fakes.shape[0], 1, grid_fakes.shape[2], grid_fakes.shape[3]])
    # atts: [b, n_latents, 1, res, res]

    for i in range(atts.shape[1]):
        att_sp = atts[:, i]  # [b, 1, x_h, x_w]
        att_sp = (att_sp - att_sp.min()) / (att_sp.max() - att_sp.min())
        grid_start_idx = i * n_samples_per
        canvas[grid_start_idx : grid_start_idx + n_samples_per] = att_sp[grid_start_idx : grid_start_idx + n_samples_per]

    # already_n_latents = 0
    # for i, att in enumerate(atts):
        # att_sp = att[-1]  # [b, n_latents, 1, x_h, x_w]
        # for j in range(att_sp.shape[1]):
            # att_sp_sub = att_sp[:, j]  # [b, 1, x_h, x_w]
            # grid_start_idx = already_n_latents * n_samples_per
            # canvas[grid_start_idx : grid_start_idx + n_samples_per] = att_sp_sub[grid_start_idx : grid_start_idx + n_samples_per]
            # already_n_latents += 1
    misc.save_image_grid(canvas,
                         filename,
                         drange=drange,
                         grid_size=grid_size)
    return

def add_outline(images, width=1):
    if images.ndim == 4:
        images[:, :, 0:width, :] = 255
        images[:, :, -width:, :] = 255
        images[:, :, :, 0:width] = 255
        images[:, :, :, -width:] = 255
    elif images.ndim == 3:
        images[:, 0:width, :] = 255
        images[:, -width:, :] = 255
        images[:, :, 0:width] = 255
        images[:, :, -width:] = 255
    else:
        raise ValueError('Unsupported dim of images:', images.shape)
    return images

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_grid_latents(n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims=None, latent_type='normal'):
    if n_discrete == 0:
        n_discrete = 1  # 0 discrete means 1 discrete
        real_has_discrete = False
    else:
        real_has_discrete = True

    grid_size = (n_samples_per, n_continuous * n_discrete)
    if latent_type == 'uniform':
        z = np.random.uniform(low=-2., high=2., size=(1, n_continuous))
    elif latent_type == 'normal':
        # z = np.random.normal(size=(1, n_continuous))
        trunc = get_truncated_normal(low=-0.5, upp=0.5)
        z = trunc.rvs(size=(1, n_continuous))
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
        # z = np.random.randn(1, n_continuous)  # [minibatch, component-3]
    grid_latents = np.tile(z, (n_continuous * n_samples_per * n_discrete, 1))
    for i in range(n_discrete):
        for j in range(n_continuous):
            # grid_latents[(i * n_continuous + j) *
                         # n_samples_per:(i * n_continuous + j + 1) *
                         # n_samples_per, j] = np.arange(
                             # -2. + 4. / float(n_samples_per+1), 2., 4. / float(n_samples_per+1))
            grid_latents[(i * n_continuous + j) *
                         n_samples_per:(i * n_continuous + j + 1) *
                         n_samples_per, j] = np.linspace(-2., 2., num=n_samples_per)
    if real_has_discrete:
        grid_discrete_ls = []
        for i in range(n_discrete):
            init_onehot = [0] * n_discrete
            init_onehot[i] = 1
            grid_discrete_ls.append(
                np.tile(np.array([init_onehot], dtype=np.float32),
                        (n_continuous * n_samples_per, 1)))
        grid_discrete = np.concatenate(grid_discrete_ls, axis=0)
        grid_latents = np.concatenate((grid_discrete, grid_latents), axis=1)
    grid_labels = np.tile(grid_labels[:1],
                          (n_discrete * n_continuous * n_samples_per, 1))
    if topk_dims is not None:
        # print('grid_latents.shape:', grid_latents.shape)
        grid_latents = np.reshape(grid_latents, [n_discrete, n_continuous, n_samples_per, -1])
        # print('grid_latents.shape:', grid_latents.shape)
        grid_latents = grid_latents[:, topk_dims]
        # print('grid_latents.shape:', grid_latents.shape)
        n_continuous = topk_dims
        grid_latents = np.reshape(grid_latents, [n_discrete * len(topk_dims) * n_samples_per, -1])
        grid_labels = np.tile(grid_labels[:1],
                              (n_discrete * len(topk_dims) * n_samples_per, 1))
        grid_size = (n_samples_per, len(topk_dims) * n_discrete)
    # grid_labels = np.tile(grid_labels[:1],
                          # (n_discrete * n_continuous * n_samples_per, 1))
    return grid_size, grid_latents, grid_labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Minibatch size.
    s.minibatch_size = minibatch_size_base
    s.minibatch_gpu = minibatch_gpu_base

    # Learning rate.
    s.G_lrate = G_lrate_base
    s.D_lrate = D_lrate_base
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(training_set.shape[1], tick_kimg_base)
    return s
