#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: plot_variation_patterns.py
# --- Creation Date: 05-08-2021
# --- Last Modified: Sat 07 Aug 2021 00:40:52 AEST
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
from scipy.linalg import expm
from training import misc
from training.utils import add_outline, get_return_v

def get_rot_matrix(identity, rnd, n_lat, scale=1):
    if identity:
        # Keep identity transformation
        return np.eye(n_lat)
    assert scale > 0
    t = rnd.uniform(-scale, scale, (n_lat, n_lat))
    t = t - t.T # Skew-symmetric matrix
    mat = expm(t) # Orthogonal matrix
    return mat

def load_pkl_from(network, load_gan):
    print('Loading networks from "%s"...' % network)

    if load_gan:
        _G, _D, I, G = misc.load_pkl(network)
    else:
        E, G = get_return_v(misc.load_pkl(network), 2)
    return G

def get_var_images(G, rnd, n_samples_per, j, latent_idx, lat_dir, eps, G_kwargs):
    z = rnd.randn(n_samples_per, *G.input_shape[1:]) # [b, nlat]
    z_var = z.copy()
    z_var[:, latent_idx] += lat_dir * eps # [b, nlat], modifying z along a latent direction on dims_of_interest
    images, atts = get_return_v(G.run(z, None, **G_kwargs), 2)  # [b, c, h, w], atts: [b, n_latents, 1, h, w]
    images_var, atts_var = get_return_v(G.run(z_var, None, **G_kwargs), 2)  # [b, c, h, w]
    return images, images_var, atts, atts_var
    
def generate_var_patterns(network, seed, latent_idx, n_perspective, n_samples_per, epsilon, thresh, scale, load_gan=False):
    '''
    Generate var patterns for multiple random latent perspectives by sampling
    latent pairs differing in orthogonal directions.
    '''
    tflib.init_tf()
    G = load_pkl_from(network, load_gan)

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.is_validation = True
    G_kwargs.randomize_noise = False
    G_kwargs.minibatch_size = 8

    rnd = np.random.RandomState(seed)

    base_dirs = np.eye(len(latent_idx)) # np of [n_var_lat, n_var_lat]
    vars_ls = []
    images_ls = []
    atts_ls = []
    for i in range(n_perspective):
        mat = get_rot_matrix(i==0, rnd, len(latent_idx), scale=scale) # np of [n_var_lat, n_var_lat]
        dirs = np.matmul(mat, base_dirs) # np of [n_var_lat, n_var_lat]
        for j, _ in enumerate(latent_idx):
            images, images_var, atts, atts_var = get_var_images(G, rnd, n_samples_per, j, latent_idx, dirs[j], epsilon, G_kwargs)
            # print('images.shape:', images.shape)
            # print('atts.shape:', atts.shape)

            # Var pattern for direction j
            var = np.abs(images - images_var).mean(axis=0).mean(axis=0, keepdims=True) # [1, h, w]
            # vars_ls.append(var / (np.max(var)) + 1e-6)
            vars_ls.append((var > thresh).astype(np.float32))

            # Example image pair
            image_pair = np.concatenate((images[0], images_var[0]), axis=1) # Concat along dim-h, [c, 2*h, w]
            images_ls.append(image_pair[np.newaxis, ...]) # [1, c, 2*h, w]

            # Atts
            if i == 0 and load_gan:
                atts = atts[:, latent_idx, 0, ...] # [b, n_lat, h, w]
                atts_ls.append(atts) # ls of [b, n_lat, h, w]

    # --- Save atts
    if load_gan: # no atts in VAEs
        atts_np = np.concatenate(atts_ls, axis=0).mean(axis=0) # [n_lat, h, w]
        # print('atts_np.shape:', atts_np.shape)
        for i, _ in enumerate(atts_np):
            atts_np[i] = atts_np[i] / (np.max(atts_np[i]) + 1e-6)
        atts_np = add_outline(atts_np, width=1)
        _, h, w = np.shape(atts_np)
        atts_np = np.reshape(atts_np, (len(latent_idx) * h, w))
        atts_np = misc.adjust_dynamic_range(atts_np, [0, 1], [0, 255])
        atts_np = np.rint(atts_np).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(atts_np, 'L').save(
            dnnlib.make_run_dir_path('att_patterns.png'))

    # --- Save variation patterns n_lat x n_perspective
    # vars_ls of len [n_perspective * len(latent_idx)]
    vars_np = np.concatenate(vars_ls, axis=0) # [n_pers*n_var_lat, h, w]
    # print('vars_np.shape:', vars_np.shape)
    vars_np = add_outline(vars_np, width=1)
    _, h, w = np.shape(vars_np)
    vars_np = np.reshape(vars_np, (n_perspective, len(latent_idx), h, w))
    vars_np = np.transpose(vars_np, [1, 2, 0, 3])
    vars_np = np.reshape(vars_np, (len(latent_idx) * h, n_perspective * w)) # [H, W]
    vars_np = misc.adjust_dynamic_range(vars_np, [0, 1], [0, 255])
    vars_np = np.rint(vars_np).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(vars_np, 'L').save(
        dnnlib.make_run_dir_path(f'var_patterns_thresh{thresh}.png'))

    # --- Save example image_pairs n_lat x n_perspective
    # images_ls of len [n_perspective * len(latent_idx)]
    pairs_np = np.concatenate(images_ls, axis=0) # [n_pers*n_var_lat, c, h, w]
    # print('pairs_np.shape:', pairs_np.shape)
    pairs_np = add_outline(pairs_np, width=1)
    _, c, h, w = np.shape(pairs_np)
    pairs_np = np.reshape(pairs_np, (n_perspective, len(latent_idx), c, h, w))
    pairs_np = np.transpose(pairs_np, [1, 3, 0, 4, 2])
    pairs_np = np.reshape(pairs_np, (len(latent_idx) * h, n_perspective * w, c)) # [H, W, c]
    pairs_np = misc.adjust_dynamic_range(pairs_np, [-1, 1] if load_gan else [0, 1], [0, 255])
    pairs_np = np.rint(pairs_np).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(pairs_np, 'RGB').save(
        dnnlib.make_run_dir_path('image_pairs.png'))

def generate_var_size_curve(network, seed, latent_idx, n_scale, max_scale, n_perspective, n_samples_per, epsilon, thresh, load_gan=False):
    '''
    Generate (var_size vs scale) curve for multiple random latent perspectives and scales by sampling
    latent pairs differing in orthogonal directions.
    '''
    tflib.init_tf()
    G = load_pkl_from(network, load_gan)

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.is_validation = True
    G_kwargs.randomize_noise = False
    G_kwargs.minibatch_size = 8

    rnd = np.random.RandomState(seed)

    base_dirs = np.eye(len(latent_idx)) # np of [n_var_lat, n_var_lat]
    max_y = 0
    min_y = 512 * 512
    fig, ax = plt.subplots()
    means_ls = []
    std_high_ls = []
    std_low_ls = []
    for scale in np.linspace(0, max_scale, n_scale):
        var_size_per_scale_ls = []
        for i in range(n_perspective if scale != 0 else 1):
            mat = get_rot_matrix(scale==0, rnd, len(latent_idx), scale=scale) # np of [n_var_lat, n_var_lat]
            dirs = np.matmul(mat, base_dirs) # np of [n_var_lat, n_var_lat]
            var_size_ls = []
            for j, _ in enumerate(latent_idx):
                images, images_var, atts, atts_var = get_var_images(G, rnd, n_samples_per, j, latent_idx, dirs[j], epsilon, G_kwargs)
                # images, atts in [b, c, h, w], [b, n_latents, 1, h, w]
                # print('images.shape:', images.shape)
                # print('atts.shape:', atts.shape)

                # Accumulate var_sizes
                var = np.abs(images - images_var).mean(axis=0).mean(axis=0) # [h, w]
                var_size = (var > thresh).astype(np.float32).sum()
                var_size_ls.append(var_size)
            var_size_per_scale_ls.append(np.array(var_size_ls).mean())
            max_y = max(max(var_size_per_scale_ls), max_y)
            min_y = min(min(var_size_per_scale_ls), min_y)
        ax.plot([scale] * len(var_size_per_scale_ls), var_size_per_scale_ls, 'ro', ms=3)

        # Get mean and std for each scale
        var_size_per_scale_np = np.array(var_size_per_scale_ls)
        var_size_per_scale_mean = var_size_per_scale_np.mean()
        var_size_per_scale_std = var_size_per_scale_np.std()
        means_ls.append(var_size_per_scale_mean)
        std_high_ls.append(var_size_per_scale_mean + var_size_per_scale_std)
        std_low_ls.append(var_size_per_scale_mean - var_size_per_scale_std)

    xticks = np.linspace(0, max_scale, n_scale)
    ax.plot(xticks, means_ls, 'g-', ms=1)
    ax.plot(xticks, std_high_ls, 'g--', ms=1)
    ax.plot(xticks, std_low_ls, 'g--', ms=1)
    ax.fill_between(xticks, std_high_ls, std_low_ls, color='green', alpha=0.3)

    ax.axis([-0.2, max_scale + 0.2, max(min_y - 50, 0), max_y + 50])
    ax.set_xlabel('Latent axis rotation scale')
    ax.set_ylabel('Varied pixels')
    ax.set_xticks(np.linspace(0, max_scale, 21))
    ax.set_xticklabels(np.linspace(0, max_scale, 21).round(1), rotation=45, ha='right')
    ax.set_yticks(np.linspace(max(min_y - 100, 0), max_y + 100, 30).round())
    print('y ticks:', np.linspace(max(min_y - 100, 0), max_y + 100, 30))
    ax.grid(True)
    # ax.show()
    ax.set_title(f'[Varied Pixels>{thresh} by Axis-Shift] vs [Axis Rotation Scales]')
    fig.savefig(dnnlib.make_run_dir_path('var_size_curve.pdf'), dpi=300)

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

    # Plot patterns
    parser_generate_var_patterns = subparsers.add_parser('generate-var-patterns', help='Generate variation patterns.')
    parser_generate_var_patterns.add_argument('--network', help='Network pickle filename', required=True)
    parser_generate_var_patterns.add_argument('--seed', type=int, help='Random seed', required=True)
    parser_generate_var_patterns.add_argument('--latent_idx', type=_str_to_list_of_int, help='latent indices to show', default='0_1_3')
    parser_generate_var_patterns.add_argument('--n_perspective', type=int, help='Number of perspectives to sample', default=5)
    parser_generate_var_patterns.add_argument('--n_samples_per', type=int, help='number of samples per sampled latent perspective', default=100)
    parser_generate_var_patterns.add_argument('--epsilon', type=float, help='Perturbation magnitude per sample pair to compute variation.', default=0.3)
    parser_generate_var_patterns.add_argument('--thresh', type=float, help='Thresh for a pixel being viewed as varied.', default=0.1)
    parser_generate_var_patterns.add_argument('--scale', type=float, help='Transformation sample scale', default=2)
    parser_generate_var_patterns.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_var_patterns.add_argument('--load_gan', help='If load GAN instead of VAE.', default=False, type=_str_to_bool)

    # Plot curve
    parser_variation_size_curve = subparsers.add_parser('generate-var-size-curve', help='Generate variation size curve.')
    parser_variation_size_curve.add_argument('--network', help='Network pickle filename', required=True)
    parser_variation_size_curve.add_argument('--seed', type=int, help='Random seed', required=True)
    parser_variation_size_curve.add_argument('--latent_idx', type=_str_to_list_of_int, help='latent indices to show', default='0_1_3')
    parser_variation_size_curve.add_argument('--n_scale', type=int, help='Number of transform scales to plot', default=20)
    parser_variation_size_curve.add_argument('--max_scale', type=float, help='The maximum of scale for plot', default=5)
    parser_variation_size_curve.add_argument('--n_perspective', type=int, help='Number of perspectives in each scale to sample', default=5)
    parser_variation_size_curve.add_argument('--n_samples_per', type=int, help='number of samples per sampled latent perspective', default=100)
    parser_variation_size_curve.add_argument('--epsilon', type=float, help='Perturbation magnitude per sample pair to compute variation.', default=0.3)
    parser_variation_size_curve.add_argument('--thresh', type=float, help='Thresh for a pixel being viewed as varied.', default=0.1)
    parser_variation_size_curve.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_variation_size_curve.add_argument('--load_gan', help='If load GAN instead of VAE.', default=False, type=_str_to_bool)

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
        'generate-var-patterns': 'plot_variation_patterns.generate_var_patterns',
        'generate-var-size-curve': 'plot_variation_patterns.generate_var_size_curve',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
