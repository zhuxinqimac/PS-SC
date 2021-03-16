#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: ps_sc_networks2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Tue 16 Mar 2021 16:27:22 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
PS-SC Networks.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from dnnlib.tflib.ops.upfirdn_2d import downsample_2d
from training.networks_stylegan2 import dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act
from training.networks_stylegan2 import minibatch_stddev_layer
from training.modular_networks2 import torgb
from training.modular_networks2 import split_module_names
from training.modular_networks2 import build_Const_layers
from training.modular_networks2 import build_C_global_layers
from training.modular_networks2 import build_noise_layer, build_conv_layer
from training.modular_networks2 import build_res_conv_layer
from training.modular_networks2 import build_C_spgroup_layers
from training.modular_networks2 import build_C_spgroup_softmax_layers

#----------------------------------------------------------------------------
# PS-SC main Generator
def G_main_ps_sc(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_ps_sc',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_modular_ps_sc',  # Build func name for the synthesis network.
        return_atts=False,  # If return atts.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_synthesis', func_name=globals()[synthesis_func], return_atts=return_atts, **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=globals()[mapping_func],
                                           dlatent_broadcast=None, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, is_training=is_training, **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        if return_atts:
            images_out, atts_out = components.synthesis.get_output_for(dlatents, is_training=is_training,
                                                             force_clean_graph=is_template_graph, return_atts=True, **kwargs)
        else:
            images_out = components.synthesis.get_output_for(dlatents, is_training=is_training,
                                                             force_clean_graph=is_template_graph, return_atts=False, **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_atts:
        atts_out = tf.identity(atts_out, name='atts_out')
        return images_out, atts_out
    else:
        return images_out


def G_mapping_ps_sc(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        latent_size=7,  # Latent vector (Z) dimensionality.
        label_size=0,  # Label dimensionality, 0 if no labels.
        mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    if label_size > 0:
        with tf.variable_scope('LabelConcat'):
            x = tf.concat([labels_in, x], axis=1)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


def G_synthesis_modular_ps_sc(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
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
        G_nf_scale=4,
        **kwargs):  # Ignore unrecognized keyword args.
    '''
    Modularized variation-consistent network2.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    act = nonlinearity
    images_out = None

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size = split_module_names(module_list)

    # Primary inputs.
    assert dlatent_size == count_dlatent_size
    dlatents_in.set_shape([None, count_dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Early layers consists of 4x4 constant layer.
    y = None

    subkwargs = EasyDict()
    subkwargs.update(dlatents_in=dlatents_in, act=act, dtype=dtype, resample_kernel=resample_kernel,
                     fused_modconv=fused_modconv, use_noise=use_noise, randomize_noise=randomize_noise,
                     resolution=resolution, fmap_base=fmap_base, architecture=architecture,
                     num_channels=num_channels,
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, **kwargs)

    # Build modules by module_dict.
    start_idx = 0
    x = dlatents_in
    atts = []
    noise_inputs = []
    for scope_idx, k in enumerate(key_ls):
        if k == 'Const':
            # e.g. {'Const': 3}
            x = build_Const_layers(init_dlatents_in=x, name=k, n_feats=size_ls[scope_idx],
                               scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
        elif k == 'C_global':
            # e.g. {'C_global': 2}
            x = build_C_global_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                      scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_spgroup-'):
            # e.g. {'C_spgroup-2': 2}
            n_subs = int(k.split('-')[-1])
            if return_atts:
                x, atts_tmp = build_C_spgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=True, n_subs=n_subs, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_spgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=False, n_subs=n_subs, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_spgroup_sm':
            # e.g. {'C_spgroup': 2}
            if return_atts:
                x, atts_tmp = build_C_spgroup_softmax_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=True, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_spgroup_softmax_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=False, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'Noise':
            # e.g. {'Noise': 1}
            # print('out noise_inputs:', noise_inputs)
            x = build_noise_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                  fmaps=nf(scope_idx//G_nf_scale), noise_inputs=noise_inputs, **subkwargs)
        elif k == 'ResConv-id' or k == 'ResConv-up' or k == 'ResConv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_res_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
        elif k == 'Conv-id' or k == 'Conv-up' or k == 'Conv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
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


#----------------------------------------------------------------------------
# Head network of infogan2.

def head_infogan2(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    images_in = fake1

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(res - 2),
                                            kernel=3,
                                            down=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t,
                                 fmaps=nf(res - 2),
                                 kernel=1,
                                 down=True,
                                 resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=dlatent_size))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x

#----------------------------------------------------------------------------
# Head network of PS-SC.


def head_ps_sc(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    images_in = fake1

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(res - 2),
                                            kernel=3,
                                            down=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t,
                                 fmaps=nf(res - 2),
                                 kernel=1,
                                 down=True,
                                 resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(dense_layer(x, fmaps=dlatent_size))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x
