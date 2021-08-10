#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_navigate.py
# --- Creation Date: 09-08-2021
# --- Last Modified: Tue 10 Aug 2021 21:32:32 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file for PS-SC related navigation networks use.
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

def get_dim_list(dims_to_learn, total_lat_in_I):
    if dims_to_learn.strip().lower() == 'all':
        return list(range(total_lat_in_I))
    return [int(x.strip()) for x in dims_to_learn[1:-1].split(',')]

def parse_config(x_type):
    '''x_type is joint by _'''
    x_ls = x_type.split('_')
    out = EasyDict()
    if x_ls[-1] == 'N':
        out.nav_type = x_ls[0]
    elif x_ls[-1] == 'I':
        out.if_train = (x_ls[0] == 'train')
    elif x_ls[-1] == 'loss':
        out.type = x_ls[0]
    return out

def run(result_dir, num_gpus, total_kimg,
        metrics, resume_pkl, G_pkl, I_pkl=None,
        I_resume_with_new_nets=False,
        I_fmap_base=8, C_lambda=1,
        I_fmap_min=16, I_fmap_max=512, G_nf_scale=4,
        n_samples_per=10, model_type='linear_N-static_I-ce_loss',
        epsilon_loss=3, random_eps=False,
        batch_size=32, batch_per_gpu=16,
        random_seed=1000, dims_to_learn='[0,1,2,3]', total_lat_in_I=20,
        learning_rate=0.002, avg_mv_for_N=False, avg_mv_for_I=False,
        use_cascade=False, cascade_alt_freq_k=1,
        network_snapshot_ticks=10):
    train = EasyDict(run_func_name='training.training_loop_nav.training_loop_nav')  # Options for training loop.
    dims_to_learn_ls = get_dim_list(dims_to_learn, total_lat_in_I)
    N_type, I_type, loss_type = model_type.split('-')
    N_configs = parse_config(N_type)
    I_configs = parse_config(I_type)
    loss_configs = parse_config(loss_type)

    # Config Nav net
    N = EasyDict(func_name='training.nav_networks.navigator', nav_type=N_configs.nav_type, n_lat=len(dims_to_learn_ls))

    # Config I net.
    I = EasyDict(pretrained_pkl=I_pkl, resume_with_new_nets=I_resume_with_new_nets, if_train=I_configs.if_train, I_fmap_base=I_fmap_base,
                 func_name='training.ps_sc_networks2.head_ps_sc', fmap_min=I_fmap_min, fmap_max=I_fmap_max, dlatent_size=total_lat_in_I)

    desc = model_type
    net_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for generator optimizer.

    # Config losses
    if loss_configs.type == 'l2':
        loss = EasyDict(func_name='training.loss_nav.nav_l2', C_lambda=C_lambda,
                        epsilon=epsilon_loss, random_eps=random_eps, dims_to_learn_ls=dims_to_learn_ls)
    else:
        raise ValueError('Not supported loss tyle: ' + loss_configs['type'])

    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': random_seed}  # Options for tflib.init_tf().

    train.total_kimg = total_kimg
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.lrate_base = learning_rate
    sched.minibatch_size_base = batch_size
    sched.minibatch_gpu_base = batch_per_gpu
    metrics = [metric_defaults[x] for x in metrics]


    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    I.fmap_base = 2 << I_fmap_base

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(N_args=N, I_args=I, net_opt_args=net_opt, loss_args=loss,
                  avg_mv_for_I=avg_mv_for_I, avg_mv_for_N=avg_mv_for_N)

    kwargs.update(sched_args=sched, grid_args=grid, metric_arg_list=metrics,
                  dims_to_learn_ls=dims_to_learn_ls,
                  tf_config=tf_config, resume_pkl=resume_pkl, G_pkl=G_pkl,
                  total_lat_in_I=total_lat_in_I, n_samples_per=n_samples_per,
                  dims_to_learn=dims_to_learn, cascade_alt_freq_k=cascade_alt_freq_k,
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
    parser = argparse.ArgumentParser(description='Train PS-SC GAN.',
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)',
                        default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)',
                        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)',
                        default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='ps_sc_gan',
                        type=str, metavar='MODEL_TYPE', choices=['gan', 'ps_sc_gan', 'info_gan', 'ps_sc_2_gan'])
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--G_pkl', help='Pretrained generator pkl.',
                        default=None, metavar='G_PKL', type=str)
    parser.add_argument('--I_pkl', help='Pretrained recognizer pkl.',
                        default=None, metavar='I_PKL', type=str)
    parser.add_argument('--I_resume_with_new_nets', help='If init a new net and use pretrained pkl.',
                        default=False, metavar='I_RESUME_WITH_NEW_NETS', type=_str_to_bool)
    parser.add_argument('--n_samples_per', help='Number of samples for each line in traversal (default: %(default)s)',
                        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--batch_size', help='N batch.',
                        metavar='N_BATCH', default=32, type=int)
    parser.add_argument('--batch_per_gpu', help='N batch per gpu.',
                        metavar='N_BATCH_PER_GPU', default=16, type=int)
    parser.add_argument('--C_lambda', help='Continuous lambda for INFO-GAN and PS-SC-GAN.',
                        metavar='C_LAMBDA', default=1, type=float)
    parser.add_argument('--epsilon_loss', help='Continuous lambda for INFO-GAN and PS-SC-GAN.',
                        metavar='EPSILON_LOSS', default=0.4, type=float)
    parser.add_argument('--random_eps', help='If use random epsilon in ps loss.',
                        default=True, metavar='RANDOM_EPS', type=_str_to_bool)
    parser.add_argument('--I_fmap_base', help='Fmap base for I.',
                        metavar='I_FMAP_BASE', default=8, type=int)
    parser.add_argument('--random_seed', help='TF random seed.',
                        metavar='RANDOM_SEED', default=9, type=int)
    parser.add_argument('--dims_to_learn', help='Dim indices in I to learn.',
                        metavar='DIMS_TO_LEARN', default='all', type=str)
    parser.add_argument('--total_lat_in_I', help='Total number of latents in I net.',
                        metavar='TOTAL_LAT_IN_I', default=20, type=int)
    parser.add_argument('--learning_rate', help='Learning rate.',
                        metavar='LEARNING_RATE', default=0.002, type=float)
    parser.add_argument('--avg_mv_for_I', help='If use average moving for I.',
                        default=False, metavar='AVG_MV_FOR_I', type=_str_to_bool)
    parser.add_argument('--avg_mv_for_N', help='If use average moving for N.',
                        default=False, metavar='AVG_MV_FOR_N', type=_str_to_bool)
    parser.add_argument('--use_cascade', help='If use cascading for PS loss.',
                        default=False, metavar='USE_CASCADE', type=_str_to_bool)
    parser.add_argument('--cascade_alt_freq_k', help='Frequency in k for cascade_dim altering.',
                        metavar='CASCADE_ALT_FREQ_K', default=1, type=float)
    parser.add_argument('--network_snapshot_ticks', help='Snapshot ticks.',
                        metavar='NETWORK_SNAPSHOT_TICKS', default=10, type=int)

    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
