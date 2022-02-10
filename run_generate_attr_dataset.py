#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_generate_attr_dataset.py
# --- Creation Date: 11-02-2022
# --- Last Modified: Fri 11 Feb 2022 05:56:14 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generate a pseudo-dataset of facial attributes using our
disentangled generator.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import collections
import cv2
import pickle
from scipy.stats import truncnorm

import pretrained_networks
from training import misc
from training.utils import get_grid_latents, get_return_v, add_outline, save_atts
from run_editing_ps_sc import image_to_ready
from run_editing_ps_sc import image_to_out
from run_generator_ps_sc import _str_to_list, _str_to_attr2idx, _str_to_bool
from PIL import Image, ImageDraw, ImageFont
from metrics.metric_defaults import metric_defaults

def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values

def generate_attr_dataset(network_pkl, n_data_samples, start_seed,
                          resolution, run_batch, used_semantics_ls, attr2idx_dict,
                          create_new_G, new_func_name, truncation_psi=0.5):
    '''
    used_semantics_ls: ['azimuth', 'haircolor', ...]
    attr2idx_dict: {'azimuth': 10, 'haircolor': 17, 'smile': 6, ...}
    '''
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    if create_new_G:
        Gs = Gs.convert(new_func_name=new_func_name)

    attr = {'names': used_semantics_ls}
    idxes = [attr2idx_dict[name] for name in used_semantics_ls]
    attr_ls = []
    for seed in range(start_seed, start_seed + n_data_samples, run_batch):
        rnd = np.random.RandomState(seed)
        if seed + run_batch >= start_seed + n_data_samples:
            b = start_seed + n_data_samples - seed
        else:
            b = run_batch
        Gs_kwargs = dnnlib.EasyDict(randomize_noise=True, minibatch_size=b, is_validation=True)
        # z = rnd.randn(b, *Gs.input_shape[1:]) # [minibatch, component]
        z = truncated_z_sample(b, Gs.input_shape[1], truncation=truncation_psi, seed=seed)
        images = get_return_v(Gs.run(z, None, **Gs_kwargs), 1) # [b, c, h, w]

        shrink = Gs.output_shape[-1] // resolution
        if shrink > 1:
            _, c, h, w = images.shape
            images = images.reshape(b, c, h // shrink, shrink, w // shrink, shrink).mean(5).mean(3)

        images = misc.adjust_dynamic_range(images, [-1, 1], [0, 255])
        images = np.transpose(images, [0, 2, 3, 1])
        images = np.rint(images).clip(0, 255).astype(np.uint8)
        for i in range(len(z)):
            PIL.Image.fromarray(images[i], 'RGB').save(dnnlib.make_run_dir_path('seed%07d.png' % (seed + i)))
        attr_ls.append(z[:, idxes])
    attr['data'] = np.concatenate(attr_ls, axis=0)
    with open(dnnlib.make_run_dir_path(f'attrs.pkl'), 'wb') as f:
        pickle.dump(attr, f)

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''Generate a pseudo-dataset of facial attributes with PS-SC GAN generator.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--n_data_samples', help='Number of data samples in the dataset', default=5000, type=int, metavar='N_DATA')
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--start_seed', help='The starting seed for generating images', default=12345, type=int)
    parser.add_argument('--resolution', help='The resolution of saved images', default=256, type=int)
    parser.add_argument('--run_batch', help='Batch size in run', default=100, type=int)
    parser.add_argument('--used_semantics_ls', help='Semantics to use', default='[azimuth, haircolor]', type=_str_to_list)
    parser.add_argument('--attr2idx_dict', help='Attr names to attr idx in latent codes',
                        default='{azimuth: 10, haircolor: 17, smile: 6}', type=_str_to_attr2idx)
    parser.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    parser.add_argument('--new_func_name', help='new G func name if create new G', default='training.ps_sc_networks2.G_main_ps_sc')
    parser.add_argument('--truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)

    args = parser.parse_args()
    kwargs = vars(args)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = 'pseudo_attr_data'

    dnnlib.submit_run(sc, 'run_generate_attr_dataset.generate_attr_dataset', **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
