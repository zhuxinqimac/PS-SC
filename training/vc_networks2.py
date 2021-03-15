#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vc_networks2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Mon 15 Mar 2021 16:44:59 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Variation-Consistency Networks2.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
from dnnlib.tflib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from training.networks_stylegan2 import get_weight, dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import naive_downsample_2d, modulated_conv2d_layer
from training.networks_stylegan2 import minibatch_stddev_layer
from training.vc_modular_networks2 import torgb
from training.vc_modular_networks2 import LATENT_MODULES
from training.vc_modular_networks2 import split_module_names
from training.vc_modular_networks2 import get_conditional_modifier
from training.vc_modular_networks2 import get_att_heat
from training.vc_modular_networks2 import build_Const_layers, build_D_layers
from training.vc_modular_networks2 import build_C_global_layers
from training.vc_modular_networks2 import build_local_heat_layers, build_local_hfeat_layers
from training.vc_modular_networks2 import build_noise_layer, build_conv_layer
from training.vc_modular_networks2 import build_res_conv_layer, build_C_fgroup_layers
from training.vc_modular_networks2 import build_C_spfgroup_layers, build_C_spgroup_layers
from training.vc_modular_networks2 import build_C_spgroup_softmax_layers
from training.vc_modular_networks2 import build_Cout_spgroup_layers
from training.vc_modular_networks2 import build_Cout_genatts_spgroup_layers
from training.vc2_subnets import build_std_gen, build_std_gen_sp
from training.networks_stylegan import instance_norm, style_mod
from training.utils import get_return_v

#----------------------------------------------------------------------------
# Variation Consistenecy main Generator
def G_main_vc2(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_vc2',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_modular_vc2',  # Build func name for the synthesis network.
        return_atts=False,  # If return atts.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_vc_synthesis', func_name=globals()[synthesis_func], return_atts=return_atts, **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_vc_mapping', func_name=globals()[mapping_func],
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


def G_mapping_vc2(
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


def G_synthesis_modular_vc2(
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
        drop_extra_torgb=False,
        latent_split_ls_for_std_gen=[5,5,5,5],  # The split list for std_gen subnets.
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
    # print('In key_ls:', key_ls)
    # print('In size_ls:', size_ls)
    # print('In count_dlatent_size:', count_dlatent_size)

    # if label_size > 0:
        # key_ls.insert(0, 'Label')
        # size_ls.insert(0, label_size)

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
                     num_channels=num_channels, latent_split_ls_for_std_gen=latent_split_ls_for_std_gen,
                     fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, **kwargs)

    # Build modules by module_dict.
    start_idx = 0
    x = dlatents_in
    atts = []
    noise_inputs = []
    # print('out 2 noise_inputs:', noise_inputs)
    for scope_idx, k in enumerate(key_ls):
        if k == 'Const':
            # e.g. {'Const': 3}
            x = build_Const_layers(init_dlatents_in=x, name=k, n_feats=size_ls[scope_idx],
                               scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
        elif k == 'D_global':
            # e.g. {'D_global': 3}
            x = build_D_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                               scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_global':
            # e.g. {'C_global': 2}
            x = build_C_global_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                      scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_fgroup':
            # e.g. {'C_fgroup': 2}
            if return_atts:
                x, atts_tmp = build_C_fgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=True, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_fgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=False, **subkwargs)
            start_idx += size_ls[scope_idx]
        # elif k == 'C_spgroup':
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
        elif k == 'C_spfgroup':
            # e.g. {'C_spfgroup': 2}
            if return_atts:
                x, atts_tmp = build_C_spfgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=True, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_C_spfgroup_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                          scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), return_atts=False, **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_local_heat':
            # e.g. {'C_local_heat': 4}
            x = build_local_heat_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                        scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k == 'C_local_hfeat':
            # e.g. {'C_local_hfeat_size': 4}
            x = build_local_hfeat_layers(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                         scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
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
        elif k == 'STD_gen':
            x = build_std_gen(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                              scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('STD_gen_sp-'):
            n_subs = int(k.split('-')[-1])
            if return_atts:
                x, atts_tmp = build_std_gen_sp(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                           scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), 
                                           return_atts=True, n_subs=n_subs, **subkwargs)
                atts.append(atts_tmp)
            else:
                x = build_std_gen_sp(x, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                           scope_idx=scope_idx, fmaps=nf(scope_idx//G_nf_scale), 
                                           return_atts=False, n_subs=n_subs, **subkwargs)
            start_idx += size_ls[scope_idx]
        else:
            raise ValueError('Unsupported module type: ' + k)

    if drop_extra_torgb:
        images_out = x
    else:
        y = torgb(x, y, num_channels=num_channels)
        images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)

    if return_atts:
        with tf.variable_scope('ConcatAtts'):
            atts_out = tf.concat(atts, axis=1)
            return tf.identity(images_out, name='images_out'), tf.identity(atts_out, name='atts_out')
    else:
        return tf.identity(images_out, name='images_out')

def att_modulated_conv2d_layer(x, y, fmaps, kernel, up=False, resample_kernel=None, resolution=128, act='lrelu'):
    with tf.variable_scope('Att_spatial'):
        x_mean = tf.reduce_mean(x, axis=[2, 3]) # [b, in_dim]
        x_wh = x.shape[2]
        n_latents = y.shape[1]
        atts_wh = dense_layer(x_mean, fmaps=n_latents * 4 * x_wh)
        atts_wh = tf.reshape(atts_wh, [-1, n_latents, 4, x_wh]) # [b, n_latents, 4, x_wh]
        att_wh_sm = tf.nn.softmax(atts_wh, axis=-1)
        att_wh_cs = tf.cumsum(att_wh_sm, axis=-1)
        att_h_cs_starts, att_h_cs_ends, att_w_cs_starts, att_w_cs_ends = tf.split(att_wh_cs, 4, axis=2)
        att_h_cs_ends = 1 - att_h_cs_ends # [b, n_latents, 1, x_wh]
        att_w_cs_ends = 1 - att_w_cs_ends # [b, n_latents, 1, x_wh]
        att_h_cs_starts = tf.reshape(att_h_cs_starts, [-1, n_latents, 1, x_wh, 1])
        att_h_cs_ends = tf.reshape(att_h_cs_ends, [-1, n_latents, 1, x_wh, 1])
        att_h = att_h_cs_starts * att_h_cs_ends # [b, n_latents, 1, x_wh, 1]
        att_w_cs_starts = tf.reshape(att_w_cs_starts, [-1, n_latents, 1, 1, x_wh])
        att_w_cs_ends = tf.reshape(att_w_cs_ends, [-1, n_latents, 1, 1, x_wh])
        att_w = att_w_cs_starts * att_w_cs_ends # [b, n_latents, 1, 1, x_wh]
        atts = att_h * att_w # [b, n_latents, 1, x_wh, x_wh]

    with tf.variable_scope('Att_apply'):
        x_norm = instance_norm(x)
        for i in range(n_latents):
            with tf.variable_scope('style_mod-' + str(i)):
                x_styled = style_mod(x_norm, y[:, i:i+1])
                x = x * (1 - atts[:, i]) + x_styled * atts[:, i]

    with tf.variable_scope('Conv_after_att'):
        # print('kernel:', kernel)
        # print('fmaps:', fmaps)
        # print('x.shape:', x.get_shape().as_list())
        # x = apply_bias_act(conv2d_layer(x, fmaps, kernel, up=up, resample_kernel=resample_kernel), act=act)
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps, kernel, up=up, resample_kernel=resample_kernel), act=act)
        with tf.variable_scope('Conv1'):
            x = conv2d_layer(x, fmaps, kernel, resample_kernel=resample_kernel)
    with tf.variable_scope('Reshape_output'):
        # print('atts.shape:', atts.get_shape().as_list())
        atts = tf.reshape(atts, [-1, x_wh, x_wh, 1])
        atts = tf.image.resize(atts, size=(resolution, resolution))
        atts = tf.reshape(atts, [-1, n_latents, 1, resolution, resolution])
    return x, atts


#----------------------------------------------------------------------------
# Variation-Consistency Head network of infogan2.


def vc2_head_infogan2(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
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
        connect_mode='concat',  # How fake1 and fake2 connected.
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
            x = apply_bias_act(dense_layer(x, fmaps=(D_global_size + (dlatent_size - D_global_size))))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x

#----------------------------------------------------------------------------
# Variation-Consistency Head network inspired by vae.


def vc2_head_byvae(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
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
            x = apply_bias_act(dense_layer(x, fmaps=(D_global_size + (dlatent_size - D_global_size))))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x


#----------------------------------------------------------------------------
# Simple I network to fit infogan.

def I_byvae_simple(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
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
        is_training=True,
        **_kwargs):  # Ignore unrecognized keyword args.

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    e1 = tf.layers.conv2d(
        inputs=fake1,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e1",
    )
    # e1 = tf.layers.batch_normalization(e1, training=is_training)
    e2 = tf.layers.conv2d(
        inputs=e1,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e2",
    )
    # e2 = tf.layers.batch_normalization(e2, training=is_training)
    e3 = tf.layers.conv2d(
        inputs=e2,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e3",
    )
    # e3 = tf.layers.batch_normalization(e3, training=is_training)
    e4 = tf.layers.conv2d(
        inputs=e3,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e4",
    )
    # e4 = tf.layers.batch_normalization(e4, training=is_training)
    with tf.variable_scope('post_encoder'):
        flat_e4 = tf.layers.flatten(e4)
        e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.leaky_relu, name="e5")
        x = tf.layers.dense(e5, dlatent_size, activation=None, name="means")
    return x


#----------------------------------------------------------------------------
# Empty VC Head network.


def vc2_empty(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        fake2,  # Second input: hidden features from z + delta(z) [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake2.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    fake2 = tf.cast(fake2, dtype)

    x = tf.zeros([fake1.shape[0], 1], dtype=fake1.dtype)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x


def D_info_modular_vc2(
        images_in,  # First input: Images [minibatch, channel, height, width].
        labels_in,  # Second input: Labels [minibatch, label_size].
        atts_in,  # Attention maps from G of fake1.
        module_D_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min= 1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1,3,3,1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        D_nf_scale=4,
        return_preds=True,
        gen_atts_in_D=False,
        no_atts_in_D=False,
        **kwargs):
    '''
    Modularized variation-consistent network2 of D.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    act = nonlinearity

    # Note that module_D_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size = split_module_names(module_D_list)
    # print('In key_ls:', key_ls)
    # print('In size_ls:', size_ls)
    # print('In count_dlatent_size:', count_dlatent_size)

    # Primary inputs.
    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    atts_in.set_shape([None, count_dlatent_size, 1, resolution, resolution])
    atts_in = atts_in[:, ::-1]
    atts_in = tf.cast(atts_in, dtype)

    subkwargs = EasyDict()
    subkwargs.update(act=act, dtype=dtype, resample_kernel=resample_kernel,
                     resolution=resolution, **kwargs)

    # Build modules by module_dict.
    x = images_in
    start_idx = 0
    len_key = len(key_ls) - 1
    pred_outs_ls = []
    gen_atts_ls = []
    for scope_idx, k in enumerate(key_ls):
        if k == 'Cout_spgroup':
            # e.g. {'Cout_spgroup': 2}
            x, pred_out = build_Cout_spgroup_layers(x, atts_in=atts_in, name=k, n_latents=size_ls[scope_idx], start_idx=start_idx,
                                      scope_idx=scope_idx, fmaps=nf((len_key - scope_idx)//D_nf_scale), **subkwargs)
            pred_outs_ls.append(pred_out) # [b, n_latents]
            start_idx += size_ls[scope_idx]
        elif k == 'Cout_genatts_spgroup':
            # e.g. {'C_spgroup': 2}
            x, pred_out, atts_tmp = build_Cout_genatts_spgroup_layers(x, name=k, n_latents=size_ls[scope_idx],
                                      scope_idx=scope_idx, fmaps=nf(scope_idx//D_nf_scale), **subkwargs)
            gen_atts_ls.append(atts_tmp)
            pred_outs_ls.append(pred_out) # [b, n_latents]
        elif k == 'ResConv-id' or k == 'ResConv-up' or k == 'ResConv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_res_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                     fmaps=nf((len_key - scope_idx)//D_nf_scale), **subkwargs)
        elif k == 'Conv-id' or k == 'Conv-up' or k == 'Conv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf((len_key - scope_idx)//D_nf_scale), **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)
    if pred_outs_ls:
        pred_outs = tf.concat(pred_outs_ls, axis=1)
    if gen_atts_ls:
        gen_atts = tf.concat(gen_atts_ls, axis=1)

    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')

    if return_preds:
        pred_outs = tf.identity(pred_outs, name='pred_outs')
        if gen_atts_in_D:
            gen_atts = tf.identity(gen_atts, name='gen_atts')
            return scores_out, pred_outs, gen_atts
        else:
            return scores_out, pred_outs
    else:
        return scores_out

def D_stylegan2_simple(
        images_in,  # First input: Images [minibatch, channel, height, width].
        labels_in,  # Second input: Labels [minibatch, label_size].
        module_D_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min= 1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1,3,3,1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        D_nf_scale=4,
        is_training=True,
        **kwargs):
    '''
    Simple D.
    '''
    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)
    e1 = tf.layers.conv2d(
        inputs=images_in,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e1",
    )
    e1 = tf.layers.batch_normalization(e1, training=is_training)
    e2 = tf.layers.conv2d(
        inputs=e1,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e2",
    )
    e2 = tf.layers.batch_normalization(e2, training=is_training)
    e3 = tf.layers.conv2d(
        inputs=e2,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e3",
    )
    e3 = tf.layers.batch_normalization(e3, training=is_training)
    e4 = tf.layers.conv2d(
        inputs=e3,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="same",
        data_format='channels_first',
        name="e4",
    )
    e4 = tf.layers.batch_normalization(e4, training=is_training)
    with tf.variable_scope('post_discriminator'):
        flat_e4 = tf.layers.flatten(e4)
        e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.leaky_relu, name="e5")
        x = tf.layers.dense(e5, 1, activation=None, name="means")
    return x


def infer_modular(
        images_in,  # First input: Images [minibatch, channel, height, width].
        dlatent_size=10,  # Number of latents to map.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min= 1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[1,3,3,1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        I_nf_scale=4,
        return_preds=True,
        gen_atts_in_D=False,
        no_atts_in_D=False,
        **kwargs):
    '''
    Modularized inference network.
    '''

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min, fmap_max)

    act = nonlinearity

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size = split_module_names(module_list)
    # print('In key_ls:', key_ls)
    # print('In size_ls:', size_ls)
    # print('In count_dlatent_size:', count_dlatent_size)

    # Primary inputs.
    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    subkwargs = EasyDict()
    subkwargs.update(act=act, dtype=dtype, resample_kernel=resample_kernel,
                     resolution=resolution, **kwargs)

    # Build modules by module_dict.
    x = images_in
    start_idx = 0
    len_key = len(key_ls) - 1
    for scope_idx, k in enumerate(key_ls):
        if k == 'ResConv-id' or k == 'ResConv-up' or k == 'ResConv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_res_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                     fmaps=nf((len_key - scope_idx)//I_nf_scale), **subkwargs)
        elif k == 'Conv-id' or k == 'Conv-up' or k == 'Conv-down':
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x, name=k, n_layers=size_ls[scope_idx], scope_idx=scope_idx,
                                 fmaps=nf((len_key - scope_idx)//I_nf_scale), **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=dlatent_size))
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')

    return scores_out
