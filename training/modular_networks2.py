#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: modular_networks2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Tue 16 Mar 2021 16:35:01 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Modular networks for PS-SC
"""

import tensorflow as tf
from training.networks_stylegan2 import dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import naive_downsample_2d, modulated_conv2d_layer
from training.networks_stylegan import instance_norm, style_mod

LATENT_MODULES = ['C_global', 'C_spgroup', 'C_spgroup_sm']

#----------------------------------------------------------------------------
# Split module list from string
def split_module_names(module_list, **kwargs):
    '''
    Split the input module_list.
    e.g. '[Const-512, Conv-up-1, C_spgroup-2-2, C_spgroup-2-2, 
    Conv-id-1, Conv-up-1, C_spgroup-2-2, C_spgroup-2-2, 
    Conv-id-1, Conv-up-1, Conv-id-1, Conv-up-1, Conv-id-1]'
    '''
    key_ls = []
    size_ls = []
    count_dlatent_size = 0
    # print('In split:', module_list)
    for module in module_list:
        m_name = module.split('-')[0]
        m_key = '-'.join(module.split('-')[:-1])  # exclude size
        size = int(module.split('-')[-1])
        if size > 0:
            if m_name in LATENT_MODULES:
                count_dlatent_size += size
            key_ls.append(m_key)
            size_ls.append(size)
    return key_ls, size_ls, count_dlatent_size

def torgb(x, y, num_channels):
    with tf.variable_scope('ToRGB'):
        t = apply_bias_act(conv2d_layer(x, fmaps=num_channels, kernel=1))
        return t if y is None else y + t

def build_Const_layers(init_dlatents_in, name, n_feats, scope_idx, dtype, **subkwargs):
    with tf.variable_scope(name + '-' + str(scope_idx)):
        x = tf.get_variable(
            'const',
            shape=[1, n_feats, 4, 4],
            initializer=tf.initializers.random_normal())
        x = tf.tile(tf.cast(x, dtype),
                    [tf.shape(init_dlatents_in)[0], 1, 1, 1])
    return x

def build_C_global_layers(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                          act, fused_modconv, fmaps=128, **kwargs):
    '''
    Build continuous latent layers, e.g. C_global layers.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        C_global_latents = dlatents_in[:, start_idx:start_idx + n_latents]
        x = apply_bias_act(modulated_conv2d_layer(x, C_global_latents, fmaps=fmaps, kernel=3,
                                                  up=False, fused_modconv=fused_modconv), act=act)
    return x

def build_C_spgroup_layers(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                          act, fused_modconv, fmaps=128, return_atts=False, resolution=128, n_subs=1, **kwargs):
    '''
    Build continuous latent layers with learned group spatial attention.
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
            C_global_latents = dlatents_in[:, start_idx:start_idx + n_latents]
            x_norm = instance_norm(x)
            for i in range(n_latents):
                with tf.variable_scope('style_mod-' + str(i)):
                    x_styled = style_mod(x_norm, C_global_latents[:, i:i+1])
                    x = x * (1 - atts[:, i]) + x_styled * atts[:, i]

        if return_atts:
            with tf.variable_scope('Reshape_output'):
                atts = tf.reshape(atts, [-1, x_wh, x_wh, 1])
                atts = tf.image.resize(atts, size=(resolution, resolution))
                atts = tf.reshape(atts, [-1, n_latents, 1, resolution, resolution])
            return x, atts
        else:
            return x

def build_C_spgroup_softmax_layers(x, name, n_latents, start_idx, scope_idx, dlatents_in,
                          act, fused_modconv, fmaps=128, return_atts=False, resolution=128, **kwargs):
    '''
    Build continuous latent layers with learned group spatial attention with pure softmax.
    Support square images only.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        with tf.variable_scope('Att_spatial'):
            x_wh = x.shape[2]
            atts = conv2d_layer(x, fmaps=n_latents, kernel=3)
            atts = tf.reshape(atts, [-1, n_latents, x_wh * x_wh]) # [b, n_latents, m]
            atts = tf.nn.softmax(atts, axis=-1)
            atts = tf.reshape(atts, [-1, n_latents, 1, x_wh, x_wh])

        with tf.variable_scope('Att_apply'):
            C_global_latents = dlatents_in[:, start_idx:start_idx + n_latents]
            x_norm = instance_norm(x)
            for i in range(n_latents):
                with tf.variable_scope('style_mod-' + str(i)):
                    x_styled = style_mod(x_norm, C_global_latents[:, i:i+1])
                    x = x * (1 - atts[:, i]) + x_styled * atts[:, i]
        if return_atts:
            with tf.variable_scope('Reshape_output'):
                atts = tf.reshape(atts, [-1, x_wh, x_wh, 1])
                atts = tf.image.resize(atts, size=(resolution, resolution))
                atts = tf.reshape(atts, [-1, n_latents, 1, resolution, resolution])
            return x, atts
        else:
            return x

def build_noise_layer(x, name, n_layers, scope_idx, act, use_noise, randomize_noise,
                      noise_inputs=None, fmaps=128, **kwargs):
    # print('in noise_inputs:', noise_inputs)
    for i in range(n_layers):
        if noise_inputs is not None:
            noise_inputs.append(tf.get_variable('noise%d' % len(noise_inputs),
                                                shape=[1, 1] + x.get_shape().as_list()[2:],
                                                initializer=tf.initializers.random_normal(),
                                                trainable=False))
        with tf.variable_scope(name + '-' + str(scope_idx) + '-' + str(i)):
            x = conv2d_layer(x, fmaps=fmaps, kernel=3, up=False)
            if use_noise:
                if randomize_noise:
                    noise = tf.random_normal(
                        [tf.shape(x)[0], 1, x.shape[2], x.shape[3]],
                        dtype=x.dtype)
                else:
                    noise = tf.cast(noise_inputs[-1], x.dtype)
                noise_strength = tf.get_variable(
                    'noise_strength-' + str(scope_idx) + '-' + str(i),
                    shape=[],
                    initializer=tf.initializers.zeros())
                x += noise * tf.cast(noise_strength, x.dtype)
            x = apply_bias_act(x, act=act)
    return x


def build_conv_layer(x,
                     name,
                     n_layers,
                     scope_idx,
                     act,
                     resample_kernel,
                     fmaps=128,
                     **kwargs):
    # e.g. {'Conv-up': 2}, {'Conv-id': 1}
    sample_type = name.split('-')[-1]
    assert sample_type in ['up', 'down', 'id']
    for i in range(n_layers):
        with tf.variable_scope(name + '-' + str(scope_idx) + '-' + str(i)):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=fmaps,
                                            kernel=3,
                                            up=(sample_type == 'up'),
                                            down=(sample_type == 'down'),
                                            resample_kernel=resample_kernel),
                               act=act)
    return x


def build_res_conv_layer(x, name, n_layers, scope_idx, act, resample_kernel, fmaps=128, **kwargs):
    # e.g. {'Conv-up': 2}, {'Conv-id': 1}
    sample_type = name.split('-')[-1]
    assert sample_type in ['up', 'down', 'id']
    x_ori = x
    for i in range(n_layers):
        with tf.variable_scope(name + '-' + str(scope_idx) + '-' + str(i)):
            x = apply_bias_act(conv2d_layer(x, fmaps=fmaps, kernel=3,
                                            up=(sample_type == 'up'),
                                            down=(sample_type == 'down'),
                                            resample_kernel=resample_kernel), act=act)
        if sample_type == 'up':
            with tf.variable_scope('Upsampling' + '-' + str(scope_idx) + '-' + str(i)):
                x_ori = naive_upsample_2d(x_ori)
        elif sample_type == 'down':
            with tf.variable_scope('Downsampling' + '-' + str(scope_idx) + '-' + str(i)):
                x_ori = naive_downsample_2d(x_ori)

    with tf.variable_scope(name + 'Resampled-' + str(scope_idx)):
        x_ori = apply_bias_act(conv2d_layer(x_ori, fmaps=fmaps, kernel=1), act=act)
        x = x + x_ori
    return x
