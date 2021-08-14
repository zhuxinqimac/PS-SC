#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_nav.py
# --- Creation Date: 09-08-2021
# --- Last Modified: Sat 14 Aug 2021 17:13:36 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Navigator training loop.
"""

import collections
import numpy as np
import pdb
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    **kwargs):  # Arguments for sub-networks (mapping and synthesis).

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Minibatch size.
    s.minibatch_size = minibatch_size_base
    s.minibatch_gpu = minibatch_gpu_base

    # Learning rate.
    s.lrate = G_lrate_base
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_base
    return s

def get_walk(w_origin, Ns, n_samples_per):
    '''
    w_origin: [1, num_ws, w_dim]
    return: [n_lat * n_samples_per, num_ws, w_dim]
    '''
    dirs = Ns.run(w_origin.mean(1)) # [n_lat, num_ws, w_dim]
    n_lat, num_ws, w_dim = dirs.shape
    step_size = 4. / n_samples_per
    w_origin = np.tile(w_origin, [n_lat, 1, 1])
    steps = []
    step = w_origin.copy()
    for i in range(n_samples_per // 2 + 1):
        step = step - i * step_size * dirs
        steps = [step[:, np.newaxis, ...]] + steps
    step = w_origin.copy()
    for i in range(1, n_samples_per - n_samples_per // 2):
        step = step + i * step_size * dirs
        steps = steps + [step[:, np.newaxis, ...]]
    steps = np.concatenate(steps, axis=1) # [n_lat, n_samples_per, num_ws, w_dim]
    return np.reshape(steps, (n_lat * n_samples_per, num_ws, w_dim))

def downsample_to_res(imgs, res=256):
    sh = imgs.shape
    if sh[2] > res:
        factor = sh[2] // res
        imgs = np.mean(np.mean(np.reshape(imgs, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=5), axis=3)
    return imgs


#----------------------------------------------------------------------------
# Main training script.

def training_loop_nav(
    N_args                  = {},       # Options for navigator network.
    I_args                  = {},       # Options for recognizer network.
    net_opt_args            = {},       # Options for global optimizer.
    loss_args               = {},       # Options for global loss.
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    # metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    N_smoothing_kimg        = 10.0,     # Half-life of the running average of navigator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    G_pkl                   = None,     # Pretrained generator pickle.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    resume_with_new_nets    = False,    # Construct new networks according to G_args and D_args before resuming training?
    n_samples_per           = 10,       # Number of samples for each line in traversal.
    avg_mv_for_I            = False,    # Use moving average in I.
    avg_mv_for_N            = False,    # Use moving average in N.
    dims_to_learn_ls        = None,     # Used dimensions in I output.
    total_lat_in_I          = None,     # Number of total latents in I.
    dims_to_learn           = None,     # Number of dimensions to learn.
    cascade_alt_freq_k      = None,     # Cascade frequency in k.
    ):

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            # G = misc.load_pkl(G_pkl) # We need to obtain G parameters num_ws, w_dim anyway.
            _, _, G = pretrained_networks.load_networks(G_pkl)
            # misc.save_pkl((G, D, I, Gs),
            N = tflib.Network('N', num_ws=G.components.mapping.static_kwargs.dlatent_broadcast,
                              w_dim=G.components.mapping.output_shape[-1], **N_args)
            Ns = N.clone('Ns')
        if I_args.pretrained_pkl is None or I_args.resume_with_new_nets:
            print('Constructing I networks...')
            I = tflib.Network('I', **I_args)

        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            rN, rG, rI, rNs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets: 
                N.copy_vars_from(rN); Ns.copy_vars_from(rNs)
            else: 
                N = rN; Ns = rNs
            if I_args.resume_with_new_nets: I.copy_vars_from(rI)
            else: I = rI
            G = rG
        elif I_args.pretrained_pkl is not None:
            print('Loading I pkl...')
            _, _, rI, _ = misc.load_pkl(I_args.pretrained_pkl)
            if I_args.resume_with_new_nets: I.copy_vars_from(rI)
            else: I = rI

    # Print layers and generate initial image snapshot.
    N.print_layers(); G.print_layers(); I.print_layers()
    # pdb.set_trace()
    sched = training_schedule(cur_nimg=total_kimg*1000, **sched_args)

    # Save traversal walk
    z_origin = np.random.normal(size=[1]+G.input_shapes[0][1:]) # [1, z_dim]
    # print('z_origin.shape:', z_origin.shape)
    _, w_origin = G.run(z_origin, None, truncation_psi=0.7, return_dlatents=True) # _, [1, num_ws, w_dim]
    # print('w_origin.shape:', w_origin.shape)
    w_walk = get_walk(w_origin, N, n_samples_per) # [n_dim * n_samples_per, num_ws, w_dim]
    # print('w_walk.shape:', w_walk.shape)
    grid_fakes = G.components.synthesis.run(w_walk, is_validation=True, minibatch_size=sched.minibatch_gpu)
    grid_fakes = downsample_to_res(grid_fakes, 256)
    grid_size = [n_samples_per, w_walk.shape[0] // n_samples_per]
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes_init.png'), drange=drange_net, grid_size=grid_size)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Ns_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), N_smoothing_kimg * 1000.0) if N_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    net_opt_args = dict(net_opt_args)
    for args in [net_opt_args]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
    net_opt = tflib.Optimizer(name='TrainG', **net_opt_args)

    # Build training graph for each GPU.
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            # Create GPU-specific shadow copies of G and D.
            N_gpu = N if gpu == 0 else N.clone(N.name + '_shadow')
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            I_gpu = I if gpu == 0 else I.clone(I.name + '_shadow')

            sched = training_schedule(cur_nimg=int(resume_kimg*1000), **sched_args)

            # Evaluate loss functions.
            with tf.name_scope('G_loss'):
                loss = dnnlib.util.call_func_by_name(N=N_gpu, G=G_gpu, I=I_gpu, opt=net_opt, minibatch_size=minibatch_gpu_in, **loss_args)

            # Register gradients.
            if I_args.if_train:
                NI_gpu_trainables = collections.OrderedDict(
                    list(N_gpu.trainables.items()) +
                    list(I_gpu.trainables.items()))
                net_opt.register_gradients(tf.reduce_mean(loss), NI_gpu_trainables)
            else:
                net_opt.register_gradients(tf.reduce_mean(loss), N_gpu.trainables)

    # Setup training ops.
    net_train_op = net_opt.apply_updates()
    Ns_update_op = Ns.setup_as_moving_average_of(N, beta=Ns_beta)

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        N.setup_weight_histograms(); G.setup_weight_histograms(); I.setup_weight_histograms()
    # metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0

        # Run training ops.
        feed_dict = {lrate_in: sched.lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([net_train_op], feed_dict)
                tflib.run([Ns_update_op], feed_dict)
            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    tflib.run(net_train_op, feed_dict)
                tflib.run(Ns_update_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 100 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                z_origin = np.random.normal(size=[1]+G.input_shapes[0][1:]) # [1, z_dim]
                _, w_origin = G.run(z_origin, None, truncation_psi=0.7, return_dlatents=True) # _, [1, num_ws, w_dim]
                w_walk = get_walk(w_origin, N, n_samples_per) # [n_dim * n_samples_per, num_ws, w_dim]
                grid_fakes = G.components.synthesis.run(w_walk, is_validation=True, minibatch_size=sched.minibatch_gpu)
                grid_fakes = downsample_to_res(grid_fakes, 256)
                grid_size = [n_samples_per, w_walk.shape[0] // n_samples_per]
                misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((N, G, I, Ns), pkl)

            # Update summaries and RunContext.
            # metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update(cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((N, G, I, Ns), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()

#----------------------------------------------------------------------------
