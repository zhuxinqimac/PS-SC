#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_ps_sc.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Tue 16 Mar 2021 17:53:00 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file for PS-SC related networks use.
Code borrowed from run_training.py from NVIDIA.
"""

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults
from training.modular_networks2 import split_module_names, LATENT_MODULES

#----------------------------------------------------------------------------


def run(dataset, data_dir, result_dir, num_gpus, total_kimg, gamma,
        mirror_augment, metrics, resume_pkl,
        I_fmap_base=8, G_fmap_base=8, D_fmap_base=9,
        fmap_decay=0.15, C_lambda=1,
        n_samples_per=10, module_list=None, model_type='ps_sc_gan',
        epsilon_loss=3, random_eps=False, latent_type='uniform',
        delta_type='onedim', batch_size=32, batch_per_gpu=16,
        return_atts=False, random_seed=1000,
        module_I_list=None, module_D_list=None,
        fmap_min=16, fmap_max=512, G_nf_scale=4,
        norm_ord=2, topk_dims_to_show=20,
        learning_rate=0.002, avg_mv_for_I=False, use_cascade=False, cascade_alt_freq_k=1,
        network_snapshot_ticks=10):
    train = EasyDict(run_func_name='training.training_loop_ps_sc.training_loop_ps_sc'
                     )  # Options for training loop.

    if not(module_list is None):
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size = split_module_names(module_list)
    if not(module_I_list is None):
        module_I_list = _str_to_list(module_I_list)
        key_I_ls, size_I_ls, count_dlatent_I_size = split_module_names(module_I_list)
    if not(module_D_list is None):
        module_D_list = _str_to_list(module_D_list)
        key_D_ls, size_D_ls, count_dlatent_D_size = split_module_names(module_D_list)

    if model_type == 'info_gan': # Independent branch version InfoGAN
        G = EasyDict(
            func_name='training.ps_sc_networks2.G_main_ps_sc',
            synthesis_func='G_synthesis_modular_ps_sc',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict(func_name='training.ps_sc_networks2.head_infogan2',
                     dlatent_size=count_dlatent_size,
                     fmap_min=fmap_min, fmap_max=fmap_max)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        desc = 'info_gan'
    elif model_type == 'ps_sc_gan': # COMA-FAIN
        G = EasyDict(
            func_name='training.ps_sc_networks2.G_main_ps_sc',
            synthesis_func='G_synthesis_modular_ps_sc',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale,
        )  # Options for generator network.
        I = EasyDict(func_name='training.ps_sc_networks2.head_ps_sc',
                     dlatent_size=count_dlatent_size,
                     fmap_min=fmap_min, fmap_max=fmap_max)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        desc = 'ps_sc_gan'
    elif model_type == 'gan': # Just modular GAN.
        G = EasyDict(
            func_name='training.ps_sc_networks2.G_main_ps_sc',
            synthesis_func='G_synthesis_modular_ps_sc',
            fmap_min=fmap_min, fmap_max=fmap_max, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size,
            module_list=module_list, use_noise=True, return_atts=return_atts,
            G_nf_scale=G_nf_scale
        )  # Options for generator network.
        I = EasyDict()
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_min=fmap_min, fmap_max=fmap_max)  # Options for discriminator network.
        desc = 'gan'
    else:
        raise ValueError('Not supported model tyle: ' + model_type)

    G_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for discriminator optimizer.
    if model_type == 'info_gan': # InfoGAN
        G_loss = EasyDict(func_name='training.loss_ps_sc.G_logistic_ns_info_gan',
            C_lambda=C_lambda,
            latent_type=latent_type, norm_ord=norm_ord)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_ps_sc.D_logistic_r1_shared',
            latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'ps_sc_gan': # PS-SC
        G_loss = EasyDict(func_name='training.loss_ps_sc.G_logistic_ns_ps_sc',
            C_lambda=C_lambda,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type,
            use_cascade=use_cascade,
            delta_type=delta_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_ps_sc.D_logistic_r1_shared',
            latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'gan': # Just GANs
        G_loss = EasyDict(func_name='training.loss_ps_sc.G_logistic_ns',
                          latent_type=latent_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss_ps_sc.D_logistic_r1_shared',
            latent_type=latent_type)  # Options for discriminator loss.

    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': random_seed}  # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = sched.D_lrate_base = learning_rate
    sched.minibatch_size_base = batch_size
    sched.minibatch_gpu_base = batch_per_gpu
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    I.fmap_base = 2 << I_fmap_base
    G.fmap_base = 2 << G_fmap_base
    D.fmap_base = 2 << D_fmap_base

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, I_args=I, G_opt_args=G_opt, D_opt_args=D_opt,
                  G_loss_args=G_loss, D_loss_args=D_loss,
                  use_info_gan=(model_type == 'info_gan'), # Independent branch version
                  use_ps_head=(model_type=='ps_sc_gan'),
                  avg_mv_for_I=avg_mv_for_I,
                  traversal_grid=True, return_atts=return_atts)
    n_continuous = 0
    if not(module_list is None):
        for i, key in enumerate(key_ls):
            m_name = key.split('-')[0]
            if (m_name in LATENT_MODULES) and (not m_name == 'D_global'):
                n_continuous += size_ls[i]

    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics,
                  tf_config=tf_config, resume_pkl=resume_pkl,
                  n_continuous=n_continuous, n_samples_per=n_samples_per,
                  topk_dims_to_show=topk_dims_to_show, cascade_alt_freq_k=cascade_alt_freq_k,
                  network_snapshot_ticks=network_snapshot_ticks)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


#----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _str_to_list(v):
    v_values = v.strip()[1:-1]
    module_list = [x.strip() for x in v_values.split(',')]
    return module_list

def _str_to_list_of_int(v):
    v_values = v.strip()[1:-1]
    step_list = [int(x.strip()) for x in v_values.split(',')]
    return step_list


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train PS-SC GAN.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--gamma',
        help='R1 regularization weight (default is config dependent)',
        default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument(
        '--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='ps_sc_gan',
                        type=str, metavar='MODEL_TYPE', choices=['gan', 'ps_sc_gan', 'info_gan'])
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--n_samples_per', help='Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--module_list', help='Module list for modular network.',
                        default=None, metavar='MODULE_LIST', type=str)
    parser.add_argument('--batch_size', help='N batch.',
                        metavar='N_BATCH', default=32, type=int)
    parser.add_argument('--batch_per_gpu', help='N batch per gpu.',
                        metavar='N_BATCH_PER_GPU', default=16, type=int)
    parser.add_argument('--C_lambda', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='C_LAMBDA', default=1, type=float)
    parser.add_argument('--epsilon_loss', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='EPSILON_LOSS', default=0.4, type=float)
    parser.add_argument('--latent_type', help='What type of latent priori to use.',
                        metavar='LATENT_TYPE', default='uniform', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument('--random_eps',
        help='If use random epsilon in ps loss.',
        default=True, metavar='RANDOM_EPS', type=_str_to_bool)
    parser.add_argument('--delta_type', help='What type of delta use.',
                        metavar='DELTA_TYPE', default='onedim', choices=['onedim', 'fulldim'], type=str)
    parser.add_argument('--fmap_decay', help='fmap decay for network building.',
                        metavar='FMAP_DECAY', default=0.15, type=float)
    parser.add_argument('--I_fmap_base', help='Fmap base for I.',
                        metavar='I_FMAP_BASE', default=8, type=int)
    parser.add_argument('--G_fmap_base', help='Fmap base for G.',
                        metavar='G_FMAP_BASE', default=8, type=int)
    parser.add_argument('--D_fmap_base', help='Fmap base for D.',
                        metavar='D_FMAP_BASE', default=9, type=int)
    parser.add_argument('--return_atts', help='If return attention maps.',
                        default=False, metavar='RETURN_ATTS', type=_str_to_bool)
    parser.add_argument('--random_seed', help='TF random seed.',
                        metavar='RANDOM_SEED', default=9, type=int)
    parser.add_argument('--module_I_list', help='Module list for I modular network.',
                        default=None, metavar='MODULE_I_LIST', type=str)
    parser.add_argument('--module_D_list', help='Module list for D modular network.',
                        default=None, metavar='MODULE_D_LIST', type=str)
    parser.add_argument('--fmap_min', help='FMAP min.',
                        metavar='FMAP_MIN', default=16, type=int)
    parser.add_argument('--fmap_max', help='FMAP max.',
                        metavar='FMAP_MAX', default=512, type=int)
    parser.add_argument('--G_nf_scale', help='N feature map scale for G.',
                        metavar='G_NF_SCALE', default=4, type=int)
    parser.add_argument('--norm_ord', help='InfoGAN loss with p-norm.',
                        metavar='NORM_ORD', default=2, type=float)
    parser.add_argument('--topk_dims_to_show', help='Number of top disentant dimensions to show in a snapshot.',
                        metavar='TOPK_DIMS_TO_SHOW', default=20, type=int)
    parser.add_argument('--learning_rate', help='Learning rate.',
                        metavar='LEARNING_RATE', default=0.002, type=float)
    parser.add_argument('--avg_mv_for_I', help='If use average moving for I.',
                        default=False, metavar='AVG_MV_FOR_I', type=_str_to_bool)
    parser.add_argument('--use_cascade', help='If use cascading for PS loss.',
                        default=False, metavar='USE_CASCADE', type=_str_to_bool)
    parser.add_argument('--cascade_alt_freq_k', help='Frequency in k for cascade_dim altering.',
                        metavar='CASCADE_ALT_FREQ_K', default=1, type=float)
    parser.add_argument('--network_snapshot_ticks', help='Snapshot ticks.',
                        metavar='NETWORK_SNAPSHOT_TICKS', default=10, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
