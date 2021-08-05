#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: plot_variation_patterns.py
# --- Creation Date: 05-08-2021
# --- Last Modified: Thu 05 Aug 2021 16:59:36 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Plot variation patterns.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import pdb
import sys
import os
import collections
import cv2

import pretrained_networks
import matplotlib.pyplot as plt
from training import misc
from training.utils import add_outline, get_return_v

def generate_var_patterns(network, seed, latent_idx, n_perspective, n_samples_per, epsilon, load_gan=False):
    '''
    Generate var patterns for multiple random latent perspectives by sampling
    latent pairs differing in orthogonal directions.
    '''
    tflib.init_tf()
    print('Loading networks from "%s"...' % network)

    if load_gan:
        _G, _D, I, G = misc.load_pkl(network)
    else:
        E, G = get_return_v(misc.load_pkl(network), 2)

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.is_validation = True
    G_kwargs.randomize_noise = False
    G_kwargs.minibatch_size = 8

    rnd = np.random.RandomState(seed)

    base_dirs = np.eye(len(latent_idx)) # np of [n_var_lat, n_var_lat]
    vars_ls = []
    for i in range(n_perspective):
        mat = get_rot_matrix(i, rnd, len(latent_idx)) # np of [n_var_lat, n_var_lat]
        dirs = np.matmul(mat, base_dirs) # np of [n_var_lat, n_var_lat]
        for j, _ in enumerate(latent_idx):
            z = rnd.randn(n_samples_per, *G.input_shape[1:]) # [b, nlat]
            z_var = z.copy()
            z_var[:, latent_idx] += dirs[j] * epsilon # [b, nlat], modifying z along a latent direction on dims_of_interest
            images, atts = get_return_v(G.run(z, None, **G_kwargs), 1)  # [b, c, h, w], atts: [b, n_latents, 1, res, res]
            images_var, atts_var = get_return_v(G.run(z_var, None, **G_kwargs), 1)  # [b, c, h, w]

            # Var pattern for direction j
            var = np.abs(images - images_var).mean(axis=0).mean(axis=0, keepdims=True) # [1, h, w]
            vars_ls.append(var)

    # vars_ls of len [n_perspective * len(latent_idx)]
    vars_np = np.concatenate(vars_ls, axis=0) # [n_pers*n_var_lat, h, w]
    _, h, w = np.shape(vars_np)
    vars_np = np.reshape(vars_np, (n_perspective, len(latent_idx), h, w))
    vars_np = np.transpose(vars_np, [0, 2, 1, 3])
    vars_np = np.reshape(vars_np, (n_perspective * h, len(latent_idx) * w)) # [H, W]
    vars_np = misc.adjust_dynamic_range(vars_np, [0, 1], [0, 255])
    vars_np = np.rint(vars_np).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(vars_np, 'L').save(
        dnnlib.make_run_dir_path('var_patterns.png'))

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2)) + 1)
    vals = s.split(',')
    return [int(x) for x in vals]


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _str_to_list_of_int(v):
    module_list = [int(x.strip()) for x in v.split('_')]
    return module_list


#----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot variation patterns.")

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_var_patterns = subparsers.add_parser('generate-var-patterns',
                                                         help='Generate grids')
    parser_generate_var_patterns.add_argument('--network',
                                              help='Network pickle filename',
                                              required=True)
    parser_generate_var_patterns.add_argument('--seed',
                                              type=int,
                                              help='Random seed',
                                              required=True)
    parser_generate_var_patterns.add_argument('--latent_idx',
                                              type=_str_to_list_of_int,
                                              help='latent indices to show',
                                              default='0_1_3')
    parser_generate_var_patterns.add_argument('--n_perspective',
                                              type=int,
                                              help='Number of perspectives to sample',
                                              default=5)
    parser_generate_var_patterns.add_argument(
        '--n_samples_per',
        type=int,
        help='number of samples per sampled latent perspective',
        default=100)
    parser_generate_var_patterns.add_argument(
        '--epsilon',
        type=float,
        help='Perturbation magnitude per sample pair to compute variation.',
        default=0.3)
    parser_generate_var_patterns.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')
    parser_generate_var_patterns.add_argument(
        '--load_gan',
        help='If load GAN instead of VAE.',
        default=False,
        type=_str_to_bool)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-var-patterns': 'plot_latent_space.generate_var_patterns',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------