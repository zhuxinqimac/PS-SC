#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: ps_sc_networks3.py
# --- Creation Date: 31-07-2021
# --- Last Modified: Wed 04 Aug 2021 23:40:33 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Updated PS-SC generator.
"""

import numpy as np
import pdb
import tensorflow as tf
from dnnlib import EasyDict
from training.modular_networks2 import torgb
from training.modular_networks2 import build_Const_layers
from training.modular_networks2 import build_C_global_layers
from training.modular_networks2 import build_noise_only_layer, build_conv_layer
from training.modular_networks2 import build_res_conv_scaled_layer
from training.modular_networks2 import build_C_spgroup_layers
from training.modular_networks2 import build_C_spgroup_softmax_layers
from training.modular_networks2 import build_C_sc_layers

#----------------------------------------------------------------------------
def G_synthesis_modular_ps_sc_2(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality.
        num_channels=1,  # Number of output color channels.
        resolution=128,  # Output resolution.
        architecture='skip', # Architecture: 'orig', 'skip', 'resnet'.
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        use_noise=False,  # If noise is used in this dataset.
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        return_atts=False,  # If return atts.
        key_ls=None,  # List of module keys.
        size_ls=None,  # List of module sizes.
        **kwargs):  # Ignore unrecognized keyword args.
    '''
    Updated modularized PS-SC generator network.
    '''
    # print('Using ps_sc_2 generator.')

    def nf(fmaps):
        return np.clip(fmaps, fmap_min, fmap_max)

    act = nonlinearity
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Early layers consists of 4x4 constant layer.
    y = None

    subkwargs = EasyDict()
    subkwargs.update(dlatents_in=dlatents_in, act=act, dtype=dtype, resample_kernel=resample_kernel,
                     fused_modconv=fused_modconv, use_noise=use_noise, randomize_noise=randomize_noise,
                     resolution=resolution, fmap_base=fmap_base, architecture=architecture,
                     num_channels=num_channels, fmap_min=fmap_min, fmap_max=fmap_max,
                     fmap_decay=fmap_decay, **kwargs)

    # Build modules by module_dict.
    start_idx = 0
    x = dlatents_in
    atts = []
    noise_inputs = []
    fmaps = fmap_base
    for scope_idx, k in enumerate(key_ls):
        if k == 'Const':
            # e.g. {'Const': 512}
            x = build_Const_layers(init_dlatents_in=x, name=k, n_feats=size_ls[scope_idx],
                                   scope_idx=scope_idx, fmaps=nf(fmaps), **subkwargs)
        elif k == 'C_global':
            # e.g. {'C_global': 2}
            x = build_C_global_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                      scope_idx=scope_idx, fmaps=nf(fmaps), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_spgroup-'):
            # e.g. {'C_spgroup-2': 2}
            n_subs = int(k.split('-')[-1])
            if return_atts:
                x, atts_tmp = build_C_spgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                                     scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=True, n_subs=n_subs, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_spgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                           scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=False, n_subs=n_subs, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_sc-'):
            # e.g. {'C_sc-mirror-2': 2}
            tokens = k.split('-')
            n_subs = int(tokens[-1])
            pre_style_dense = ('prestyle' in tokens)
            mirrored_masks = ('mirror' in tokens)
            if return_atts:
                x, atts_tmp = build_C_sc_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                                      scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=True,
                                                      n_subs=n_subs, mirrored_masks=mirrored_masks,
                                                      pre_style_dense=pre_style_dense, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_sc_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                            scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=False,
                                            n_subs=n_subs, mirrored_masks=mirrored_masks,
                                            pre_style_dense=pre_style_dense, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_spgroup_sm':
            # e.g. {'C_spgroup_sm': 2}
            if return_atts:
                x, atts_tmp = build_C_spgroup_softmax_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                                             scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=True, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_spgroup_softmax_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                                   scope_idx=scope_idx, fmaps=nf(fmaps), return_atts=False, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'Noise':
            # e.g. {'Noise': 1}
            # print('out noise_inputs:', noise_inputs)
            x = build_noise_only_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                       noise_inputs=noise_inputs, **subkwargs)
        elif k in ('ResConv-id', 'ResConv-up', 'ResConv-down'):
            # e.g. {'ResConv-up': 2}, {'ResConv-id': 1}
            if k == 'ResConv-up':
                fmaps = int(fmaps / 2.0)
            elif k == 'ResConv-downw':
                fmaps = int(fmaps * 2.0)
            x = build_res_conv_scaled_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                            fmaps=nf(fmaps), **subkwargs)
        elif k in ('Conv-id', 'Conv-up', 'Conv-down'):
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            if k == 'Conv-up':
                fmaps = int(fmaps / 2.0)
            elif k == 'Conv-downw':
                fmaps = int(fmaps * 2.0)
            x = build_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf(fmaps), **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)

    if return_atts:
        with tf.variable_scope('ConcatAtts'):
            atts_out = tf.concat(atts, axis=1)
            return tf.identity(images_out, name='images_out'), tf.identity(atts_out, name='atts_out')
    else:
        return tf.identity(images_out, name='images_out')
