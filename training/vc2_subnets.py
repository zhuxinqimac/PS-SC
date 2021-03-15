#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: vc2_subnets.py
# --- Creation Date: 11-10-2020
# --- Last Modified: Thu 11 Mar 2021 23:11:38 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Modular networks for VC2
"""

import numpy as np
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
from training.networks_stylegan import instance_norm, style_mod
from training.utils import get_return_v

def build_std_gen(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                  act, fused_modconv, fmaps=128, resolution=512, fmap_base=2 << 8,
                  fmap_min=1, fmap_max=512, fmap_decay=1,
                  architecture='skip', randomize_noise=True,
                  resample_kernel=[1,3,3,1], num_channels=3,
                  latent_split_ls_for_std_gen=[5,5,5,5],
                  **kwargs):
    '''
    Build standard disentanglement generator with similar architecture to stylegan2.
    '''
    # with tf.variable_scope(name + '-' + str(scope_idx)):
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    num_layers = resolution_log2 * 2 - 2
    images_out = None
    dtype = x.dtype
    assert n_latents == sum(latent_split_ls_for_std_gen)
    assert num_layers == len(latent_split_ls_for_std_gen)
    latents_ready_ls = []
    start_code = 0
    for i, seg in enumerate(latent_split_ls_for_std_gen):
        with tf.variable_scope('PreConvDense-' + str(i) + '-0'):
            x_tmp0 = dense_layer(dlatents_in[:, start_code:start_code+seg], fmaps=nf(1))
        with tf.variable_scope('PreConvDense-' + str(i) + '-1'):
            x_tmp1 = dense_layer(x_tmp0, fmaps=nf(1))
        start_code += seg
        latents_ready_ls.append(x_tmp1)

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        # start_idx_layer = sum(latent_split_ls_for_std_gen[:layer_idx])
        # for i in range(start_idx_layer, start_idx_layer + latent_split_ls_for_std_gen[layer_idx]):
            # x = modulated_conv2d_layer(x, latents_ready_spl_ls[i], fmaps=fmaps, kernel=kernel, up=up,
                                       # resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        x = modulated_conv2d_layer(x, latents_ready_ls[layer_idx], fmaps=fmaps, kernel=kernel, up=up,
                                   resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res): # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)
    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(modulated_conv2d_layer(x, latents_ready_ls[res*2-3], fmaps=num_channels, kernel=1,
                                                      demodulate=False, fused_modconv=fused_modconv))
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)

    # Main layers.
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(x, res)
            if res == resolution_log2:
                y = torgb(x, y, res)
    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')

def build_std_gen_sp(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                     act, fused_modconv, fmaps=128, resolution=512, fmap_base=2 << 8,
                     fmap_min=1, fmap_max=512, fmap_decay=1,
                     architecture='skip', randomize_noise=True,
                     resample_kernel=[1,3,3,1], num_channels=3,
                     latent_split_ls_for_std_gen=[5,5,5,5],
                     n_subs=4, return_atts=True,
                     **kwargs):
    '''
    Build standard disentanglement generator with similar architecture to stylegan2.
    '''
    # with tf.variable_scope(name + '-' + str(scope_idx)):
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    num_layers = resolution_log2 * 2 - 2
    images_out = None
    dtype = x.dtype
    assert n_latents == sum(latent_split_ls_for_std_gen)
    assert num_layers == len(latent_split_ls_for_std_gen)
    latents_ready_spl_ls = []
    for i in range(n_latents):
        with tf.variable_scope('PreConvDense-' + str(i) + '-0'):
            x_tmp0 = dense_layer(dlatents_in[:, i:i+1], fmaps=nf(1))
        with tf.variable_scope('PreConvDense-' + str(i) + '-1'):
            x_tmp1 = dense_layer(x_tmp0, fmaps=nf(1))
        latents_ready_spl_ls.append(x_tmp1[:, tf.newaxis, ...])

    latents_ready_ls = []
    start_code = 0
    for i, seg in enumerate(latent_split_ls_for_std_gen):
        with tf.variable_scope('PreConvConcat-' + str(i)):
            x_tmp = tf.concat(latents_ready_spl_ls[start_code:start_code+seg], axis=1)
        latents_ready_ls.append(x_tmp)
        start_code += seg

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 1):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Single convolution layer with all the bells and whistles.
    def layer(x, layer_idx, fmaps, kernel, up=False):
        x, atts = get_return_v(build_C_spgroup_layers_with_latents_ready(x, 'SP_latents', latent_split_ls_for_std_gen[layer_idx],
                                                                         layer_idx, latents_ready_ls[layer_idx], return_atts=return_atts,
                                                                         resolution=resolution, n_subs=n_subs, **kwargs), 2)
        x = conv2d_layer(x, fmaps=fmaps, kernel=kernel, up=up)
        if randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_inputs[layer_idx], x.dtype)
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act), atts

    # Building blocks for main layers.
    def block(x, res): # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x, atts_0 = layer(x, layer_idx=res*2-5, fmaps=nf(res-1), kernel=3, up=True)
        with tf.variable_scope('Conv1'):
            x, atts_1 = layer(x, layer_idx=res*2-4, fmaps=nf(res-1), kernel=3)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        if return_atts:
            atts = tf.concat([atts_0, atts_1], axis=1)
        else:
            atts = None
        return x, atts
    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)
    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            # t = apply_bias_act(modulated_conv2d_layer(x, latents_ready_ls[res*2-3], fmaps=num_channels, kernel=1,
                                                      # demodulate=False, fused_modconv=fused_modconv))
            t, atts = get_return_v(build_C_spgroup_layers_with_latents_ready(x, 'SP_latents', latent_split_ls_for_std_gen[res*2-3],
                                                                             res*2-3, latents_ready_ls[res*2-3], return_atts=return_atts,
                                                                             resolution=resolution, n_subs=n_subs, **kwargs), 2)
            t = apply_bias_act(conv2d_layer(t, fmaps=num_channels, kernel=1))
            return t if y is None else y + t, atts

    # Early layers.
    y = None
    atts_out_ls = []
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x, atts_tmp = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)
            atts_out_ls.append(atts_tmp)

    # Main layers.
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x, atts_tmp = block(x, res)
            atts_out_ls.append(atts_tmp)
            if res == resolution_log2:
                y, atts_tmp_final = torgb(x, y, res)
                atts_out_ls.append(atts_tmp_final)
    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)

    if return_atts:
        with tf.variable_scope('ConcatAtts'):
            atts_out = tf.concat(atts_out_ls, axis=1)
            return tf.identity(images_out, name='images_out'), tf.identity(atts_out, name='atts_out')
    else:
        return tf.identity(images_out, name='images_out')

def build_C_spgroup_layers_with_latents_ready(x, name, n_latents, scope_idx, latents_ready,
                                              return_atts=False, resolution=128, n_subs=1, **kwargs):
    '''
    Build continuous latent layers with learned group spatial attention using latents_ready.
    Support square images only.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        with tf.variable_scope('Att_spatial'):
            x_mean = tf.reduce_mean(x, axis=[2, 3]) # [b, in_dim]
            x_wh = x.shape[2]
            atts_wh = dense_layer(x_mean, fmaps=n_latents * n_subs * 4 * x_wh)
            atts_wh = tf.reshape(atts_wh, [-1, n_latents, n_subs, 4, x_wh]) # [b, n_latents, n_subs, 4, x_wh]
            att_wh_sm = tf.nn.softmax(atts_wh, axis=-1)
            att_wh_cs = tf.cumsum(att_wh_sm, axis=-1)
            att_h_cs_starts, att_h_cs_ends, att_w_cs_starts, att_w_cs_ends = tf.split(att_wh_cs, 4, axis=3)
            att_h_cs_ends = 1 - att_h_cs_ends # [b, n_latents, n_subs, 1, x_wh]
            att_w_cs_ends = 1 - att_w_cs_ends # [b, n_latents, n_subs, 1, x_wh]
            att_h_cs_starts = tf.reshape(att_h_cs_starts, [-1, n_latents, n_subs, 1, x_wh, 1])
            att_h_cs_ends = tf.reshape(att_h_cs_ends, [-1, n_latents, n_subs, 1, x_wh, 1])
            att_h = att_h_cs_starts * att_h_cs_ends # [b, n_latents, n_subs, 1, x_wh, 1]
            att_w_cs_starts = tf.reshape(att_w_cs_starts, [-1, n_latents, n_subs, 1, 1, x_wh])
            att_w_cs_ends = tf.reshape(att_w_cs_ends, [-1, n_latents, n_subs, 1, 1, x_wh])
            att_w = att_w_cs_starts * att_w_cs_ends # [b, n_latents, n_subs, 1, 1, x_wh]
            atts = att_h * att_w # [b, n_latents, n_subs, 1, x_wh, x_wh]
            atts = tf.reduce_mean(atts, axis=2) # [b, n_latents, 1, x_wh, x_wh]
            # atts = tf.reduce_sum(atts, axis=2) # [b, n_latents, 1, x_wh, x_wh]

        with tf.variable_scope('Att_apply'):
            C_global_latents = latents_ready # [b, n_latents, 512]
            x_norm = instance_norm(x)
            # x_norm = tf.tile(x_norm, [1, n_latents, 1, 1])
            # x_norm = tf.reshape(x_norm, [-1, x.shape[1], x.shape[2], x.shape[3]]) # [b*n_latents, c, h, w]
            # C_global_latents = tf.reshape(C_global_latents, [-1, 1])
            # x_styled = style_mod(x_norm, C_global_latents)
            # x_styled = tf.reshape(x_styled, [-1, n_latents, x_styled.shape[1],
                                             # x_styled.shape[2], x_styled.shape[3]])
            for i in range(n_latents):
                with tf.variable_scope('style_mod-' + str(i)):
                    x_styled = style_mod(x_norm, C_global_latents[:, i])
                    x = x * (1 - atts[:, i]) + x_styled * atts[:, i]
                    # x = x * (1 - atts[:, i]) + x_styled[:, i] * atts[:, i]

        if return_atts:
            with tf.variable_scope('Reshape_output'):
                atts = tf.reshape(atts, [-1, x_wh, x_wh, 1])
                atts = tf.image.resize(atts, size=(resolution, resolution))
                atts = tf.reshape(atts, [-1, n_latents, 1, resolution, resolution])
            return x, atts
        else:
            return x
