#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: training_loop_vc2.py
# --- Creation Date: 24-04-2020
# --- Last Modified: Mon 15 Mar 2021 18:10:30 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Training loop file for variation consistency (V2) related networks.
Code borrowed from training_loop.py of NVIDIA.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base
from training.training_loop import process_reals, training_schedule
from training.utils import save_atts, add_outline, get_grid_latents

#----------------------------------------------------------------------------
# Main training script.


def training_loop_vc2(
        G_args={},  # Options for generator network.
        D_args={},  # Options for discriminator network.
        I_args={},  # Options for infogan-head/vcgan-head network.
        I_info_args={},  # Options for infogan-head/vcgan-head network.
        G_opt_args={},  # Options for generator optimizer.
        D_opt_args={},  # Options for discriminator optimizer.
        G_loss_args={},  # Options for generator loss.
        D_loss_args={},  # Options for discriminator loss.
        dataset_args={},  # Options for dataset.load_dataset().
        sched_args={},  # Options for train.TrainingSchedule.
        grid_args={},  # Options for train.setup_snapshot_image_grid().
        metric_arg_list=[],  # Options for MetricGroup.
        tf_config={},  # Options for tflib.init_tf().
        use_info_gan=False,  # Whether to use info-gan.
        use_vc_head=False,  # Whether to use vc-head.
        data_dir=None,  # Directory to load datasets from.
        G_smoothing_kimg=10.0,  # Half-life of the running average of generator weights.
        minibatch_repeats=4,  # Number of minibatches to run before adjusting training parameters.
        lazy_regularization=True,  # Perform regularization as a separate training step?
        G_reg_interval=4,  # How often the perform regularization for G? Ignored if lazy_regularization=False.
        D_reg_interval=16,  # How often the perform regularization for D? Ignored if lazy_regularization=False.
        reset_opt_for_new_lod=True,  # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        mirror_augment=False,  # Enable mirror augment?
        drange_net=[
            -1, 1
        ],  # Dynamic range used when feeding image data to the networks.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = only save 'networks-final.pkl'.
        save_tf_graph=False,  # Include full TensorFlow computation graph in the tfevents file?
        save_weight_histograms=False,  # Include weight histograms in the tfevents file?
        resume_pkl=None,  # Network pickle to resume training from, None = train from scratch.
        resume_kimg=0.0,  # Assumed training progress at the beginning. Affects reporting and training schedule.
        resume_time=0.0,  # Assumed wallclock time at the beginning. Affects reporting.
        resume_with_new_nets=False,  # Construct new networks according to G_args and D_args before resuming training?
        traversal_grid=False,  # Used for disentangled representation learning.
        n_discrete=3,  # Number of discrete latents in model.
        n_continuous=4,  # Number of continuous latents in model.
        return_atts=False,  # If return attention maps.
        return_I_atts=False,  # If return I_attention maps of vpex.
        avg_mv_for_I=False,  # If use average moving for I.
        opt_reset_ls=None,  # Reset lr list for gradual latents.
        topk_dims_to_show=20, # Number of top disentant dimensions to show in a snapshot.
        cascade_alt_freq_k=1, # Frequency in k for cascade_dim altering.
        n_samples_per=10):  # Number of samples for each line in traversal.

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # If include I
    include_I = use_info_gan or use_vc_head

    # Load training set.
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir),
                                        verbose=True,
                                        **dataset_args)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(
        training_set, **grid_args)
    grid_fakes = add_outline(grid_reals, width=1)
    misc.save_image_grid(grid_reals,
                         dnnlib.make_run_dir_path('reals.png'),
                         drange=training_set.dynamic_range,
                         grid_size=grid_size)

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            print('G_args:', G_args)
            G = tflib.Network('G',
                              num_channels=training_set.shape[0],
                              resolution=training_set.shape[1],
                              label_size=training_set.label_size,
                              **G_args)
            D = tflib.Network('D',
                              num_channels=training_set.shape[0],
                              resolution=training_set.shape[1],
                              label_size=training_set.label_size,
                              **D_args)
            if include_I:
                I = tflib.Network('I',
                                  num_channels=training_set.shape[0],
                                  resolution=training_set.shape[1],
                                  label_size=training_set.label_size,
                                  **I_args)
                if avg_mv_for_I:
                    Is = I.clone('Is')
            
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            if include_I:
                if avg_mv_for_I:
                    rG, rD, rI, rGs, rIs = misc.load_pkl(resume_pkl)
                else:
                    rG, rD, rI, rGs = misc.load_pkl(resume_pkl)
            else:
                rG, rD, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets:
                G.copy_vars_from(rG)
                D.copy_vars_from(rD)
                if include_I:
                    I.copy_vars_from(rI)
                    if avg_mv_for_I:
                        Is.copy_vars_from(rIs)
                Gs.copy_vars_from(rGs)
            else:
                G = rG
                D = rD
                if include_I:
                    I = rI
                    if avg_mv_for_I:
                        Is = rIs
                Gs = rGs

    # Print layers and generate initial image snapshot.
    G.print_layers()
    D.print_layers()
    if include_I:
        I.print_layers()
    sched = training_schedule(cur_nimg=total_kimg * 1000,
                              training_set=training_set,
                              **sched_args)
    if traversal_grid:
        if topk_dims_to_show > 0:
            topk_dims = np.arange(min(topk_dims_to_show, n_continuous))
        else:
            topk_dims = np.arange(n_continuous)
        print('topk_dims_to_show:', topk_dims_to_show)
        grid_size, grid_latents, grid_labels = get_grid_latents(
            n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims)
    else:
        grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    print('grid_size:', grid_size)
    print('grid_latents.shape:', grid_latents.shape)
    print('grid_labels.shape:', grid_labels.shape)
    if return_atts:
        grid_fakes, atts = Gs.run(grid_latents,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=sched.minibatch_gpu,
                            randomize_noise=True,
                            return_atts=True,
                            resolution=training_set.shape[1])
        # atts: [b, n_latents, 1, res, res]
        atts = atts[:, topk_dims]
        save_atts(atts,
                  filename=dnnlib.make_run_dir_path('fakes_atts_init.png'),
                  grid_size=grid_size,
                  drange=[0, 1],
                  grid_fakes=grid_fakes,
                  n_samples_per=n_samples_per)
    else:
        grid_fakes = Gs.run(grid_latents,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=sched.minibatch_gpu,
                            randomize_noise=True)
    grid_fakes = add_outline(grid_fakes, width=1)
    misc.save_image_grid(grid_fakes,
                         dnnlib.make_run_dir_path('fakes_init.png'),
                         drange=drange_net,
                         grid_size=grid_size)

    if include_I and return_I_atts:
        if avg_mv_for_I:
            I_tmp = Is
        else:
            I_tmp = I
        _, atts = I_tmp.run(grid_fakes,
                            grid_fakes,
                            grid_latents,
                            is_validation=True,
                            minibatch_size=sched.minibatch_gpu,
                            return_atts=True,
                            resolution=training_set.shape[1])
        save_atts(atts,
                  filename=dnnlib.make_run_dir_path('fakes_I_atts_init.png'),
                  grid_size=grid_size,
                  drange=[0, 1],
                  grid_fakes=grid_fakes,
                  n_samples_per=n_samples_per)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in = tf.placeholder(tf.int32,
                                           name='minibatch_size_in',
                                           shape=[])
        minibatch_gpu_in = tf.placeholder(tf.int32,
                                          name='minibatch_gpu_in',
                                          shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in *
                                                     num_gpus)
        Gs_beta = 0.5**tf.div(tf.cast(minibatch_size_in,
                                      tf.float32), G_smoothing_kimg *
                              1000.0) if G_smoothing_kimg > 0.0 else 0.0
        cascade_dim = tf.placeholder(tf.int32, name='cascade_dim', shape=[])

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for args, reg_interval in [(G_opt_args, G_reg_interval),
                               (D_opt_args, D_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)

    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            if include_I:
                I_gpu = I if gpu == 0 else I.clone(I.name + '_shadow')

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'):
                sched = training_schedule(cur_nimg=int(resume_kimg * 1000),
                                          training_set=training_set,
                                          **sched_args)
                reals_var = tf.Variable(
                    name='reals',
                    trainable=False,
                    initial_value=tf.zeros([sched.minibatch_gpu] +
                                           training_set.shape))
                labels_var = tf.Variable(name='labels',
                                         trainable=False,
                                         initial_value=tf.zeros([
                                             sched.minibatch_gpu,
                                             training_set.label_size
                                         ]))
                reals_write, labels_write = training_set.get_minibatch_tf()
                reals_write, labels_write = process_reals(
                    reals_write, labels_write, lod_in, mirror_augment,
                    training_set.dynamic_range, drange_net)
                reals_write = tf.concat(
                    [reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                labels_write = tf.concat(
                    [labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.assign(reals_var, reals_write)]
                data_fetch_ops += [tf.assign(labels_var, labels_write)]
                reals_read = reals_var[:minibatch_gpu_in]
                labels_read = labels_var[:minibatch_gpu_in]

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars:
                lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars:
                lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    if include_I:
                        G_loss, G_reg = dnnlib.util.call_func_by_name(
                            G=G_gpu, D=D_gpu, I=I_gpu,
                            opt=G_opt, training_set=training_set,
                            minibatch_size=minibatch_gpu_in,
                            cascade_dim=cascade_dim,
                            **G_loss_args)
                    else:
                        G_loss, G_reg = dnnlib.util.call_func_by_name(
                            G=G_gpu, D=D_gpu,
                            opt=G_opt, training_set=training_set,
                            minibatch_size=minibatch_gpu_in,
                            **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(
                        G=G_gpu, D=D_gpu, opt=D_opt,
                        training_set=training_set,
                        minibatch_size=minibatch_gpu_in,
                        reals=reals_read,
                        labels=labels_read,
                        **D_loss_args)

            # Register gradients.
            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None:
                    G_reg_opt.register_gradients(
                        tf.reduce_mean(G_reg * G_reg_interval),
                        G_gpu.trainables)
                if D_reg is not None:
                    D_reg_opt.register_gradients(
                        tf.reduce_mean(D_reg * D_reg_interval),
                        D_gpu.trainables)
            if include_I:
                GI_gpu_trainables = collections.OrderedDict(
                    list(G_gpu.trainables.items()) +
                    list(I_gpu.trainables.items()))
                G_opt.register_gradients(tf.reduce_mean(G_loss),
                                         GI_gpu_trainables)
                D_opt.register_gradients(tf.reduce_mean(D_loss),
                                         D_gpu.trainables)
            else:
                G_opt.register_gradients(tf.reduce_mean(G_loss),
                                         G_gpu.trainables)
                D_opt.register_gradients(tf.reduce_mean(D_loss),
                                         D_gpu.trainables)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    if avg_mv_for_I:
        Is_update_op = Is.setup_as_moving_average_of(I, beta=Gs_beta)

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
        G.setup_weight_histograms()
        D.setup_weight_histograms()
        if include_I:
            I.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('',
                                   cur_epoch=resume_kimg,
                                   max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg,
                                  training_set=training_set,
                                  **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(
                    sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state()
                D_opt.reset_optimizer_state()
        # if opt_reset_ls is not None:
            # if cur_nimg in opt_reset_ls:
                # G_opt.reset_optimizer_state()
                # D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Calculate which cascade_dim is to use.
        cur_nimg_k = cur_nimg // int(cascade_alt_freq_k * 1000)
        sched_cascade_dim = cur_nimg_k % n_continuous

        # Run training ops.
        feed_dict = {
            lod_in: sched.lod,
            lrate_in: sched.G_lrate,
            minibatch_size_in: sched.minibatch_size,
            minibatch_gpu_in: sched.minibatch_gpu,
            cascade_dim: sched_cascade_dim
        }
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size,
                           sched.minibatch_gpu * num_gpus)
            run_G_reg = (lazy_regularization
                         and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization
                         and running_mb_counter % D_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([G_train_op, data_fetch_op], feed_dict)
                if run_G_reg:
                    tflib.run(G_reg_op, feed_dict)
                if avg_mv_for_I:
                    tflib.run([D_train_op, Gs_update_op, Is_update_op], feed_dict)
                else:
                    tflib.run([D_train_op, Gs_update_op], feed_dict)
                if run_D_reg:
                    tflib.run(D_reg_op, feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    tflib.run(G_train_op, feed_dict)
                if run_G_reg:
                    for _round in rounds:
                        tflib.run(G_reg_op, feed_dict)
                if avg_mv_for_I:
                    tflib.run([Gs_update_op, Is_update_op], feed_dict)
                else:
                    tflib.run(Gs_update_op, feed_dict)
                for _round in rounds:
                    tflib.run(data_fetch_op, feed_dict)
                    tflib.run(D_train_op, feed_dict)
                if run_D_reg:
                    for _round in rounds:
                        tflib.run(D_reg_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start(
            ) + resume_time

            # Report progress.
            print(
                'tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f'
                % (autosummary('Progress/tick', cur_tick),
                   autosummary('Progress/kimg', cur_nimg / 1000.0),
                   autosummary('Progress/lod', sched.lod),
                   autosummary('Progress/minibatch', sched.minibatch_size),
                   dnnlib.util.format_time(
                       autosummary('Timing/total_sec', total_time)),
                   autosummary('Timing/sec_per_tick', tick_time),
                   autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                   autosummary('Timing/maintenance_sec', maintenance_time),
                   autosummary('Resources/peak_gpu_mem_gb',
                               peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if network_snapshot_ticks is not None and (
                    cur_tick % network_snapshot_ticks == 0 or done):
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' %
                                               (cur_nimg // 1000))
                if include_I:
                    if avg_mv_for_I:
                        misc.save_pkl((G, D, I, Gs, Is), pkl)
                    else:
                        misc.save_pkl((G, D, I, Gs), pkl)
                else:
                    misc.save_pkl((G, D, Gs), pkl)
                met_outs = metrics.run(pkl,
                                       run_dir=dnnlib.make_run_dir_path(),
                                       data_dir=dnnlib.convert_path(data_dir),
                                       num_gpus=num_gpus,
                                       tf_config=tf_config,
                                       include_I=include_I,
                                       avg_mv_for_I=avg_mv_for_I,
                                       Gs_kwargs=dict(is_validation=True, return_atts=False),
                                       mapping_nodup=True)
                if topk_dims_to_show > 0:
                    if 'tpl_per_dim' in met_outs:
                        avg_distance_per_dim = met_outs['tpl_per_dim'] # shape: (n_continuous)
                        topk_dims = np.argsort(avg_distance_per_dim)[::-1][:topk_dims_to_show] # shape: (20)
                    else:
                        topk_dims = np.arange(min(topk_dims_to_show, n_continuous))
                else:
                    topk_dims = np.arange(n_continuous)

            if image_snapshot_ticks is not None and (
                    cur_tick % image_snapshot_ticks == 0 or done):
                if traversal_grid:
                    grid_size, grid_latents, grid_labels = get_grid_latents(
                        n_discrete, n_continuous, n_samples_per, G, grid_labels, topk_dims)
                else:
                    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])

                if return_atts:
                    grid_fakes, atts = Gs.run(grid_latents,
                                        grid_labels,
                                        is_validation=True,
                                        minibatch_size=sched.minibatch_gpu,
                                        randomize_noise=True,
                                        return_atts=True,
                                        resolution=training_set.shape[1])
                    # atts: [b, n_latents, 1, res, res]
                    atts = atts[:, topk_dims]
                    save_atts(atts,
                              filename=dnnlib.make_run_dir_path('fakes_atts%06d.png' % (cur_nimg // 1000)),
                              grid_size=grid_size,
                              drange=[0, 1],
                              grid_fakes=grid_fakes,
                              n_samples_per=n_samples_per)
                else:
                    grid_fakes = Gs.run(grid_latents,
                                        grid_labels,
                                        is_validation=True,
                                        minibatch_size=sched.minibatch_gpu,
                                        randomize_noise=True)
                grid_fakes = add_outline(grid_fakes, width=1)
                misc.save_image_grid(grid_fakes,
                                     dnnlib.make_run_dir_path(
                                         'fakes%06d.png' % (cur_nimg // 1000)),
                                     drange=drange_net,
                                     grid_size=grid_size)
                if include_I and return_I_atts:
                    if avg_mv_for_I:
                        I_tmp = Is
                    else:
                        I_tmp = I
                    _, atts = I_tmp.run(grid_fakes,
                                        grid_fakes,
                                        grid_latents,
                                        is_validation=True,
                                        minibatch_size=sched.minibatch_gpu,
                                        return_atts=True,
                                        resolution=training_set.shape[1])
                    atts = atts[:, topk_dims]
                    save_atts(atts,
                              filename=dnnlib.make_run_dir_path('fakes_I_atts%06d.png' % (cur_nimg // 1000)),
                              grid_size=grid_size,
                              drange=[0, 1],
                              grid_fakes=grid_fakes,
                              n_samples_per=n_samples_per)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod,
                                           cur_epoch=cur_nimg // 1000,
                                           max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get(
            ).get_last_update_interval() - tick_time

    # Save final snapshot.
    if include_I:
        if avg_mv_for_I:
            misc.save_pkl((G, D, I, Gs, Is),
                          dnnlib.make_run_dir_path('network-final.pkl'))
        else:
            misc.save_pkl((G, D, I, Gs),
                          dnnlib.make_run_dir_path('network-final.pkl'))
    else:
        misc.save_pkl((G, D, Gs),
                      dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()


#----------------------------------------------------------------------------
