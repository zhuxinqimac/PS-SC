# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014

def G_logistic(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -tf.nn.softplus(fake_scores_out) # log(1-sigmoid(fake_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

def G_logistic_ns(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def G_logistic_ns_dsp(G, D, opt, training_set, minibatch_size, latent_type='uniform', D_global_size=0):
    _ = opt
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, _ = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    return loss, None

def calc_info_gan_loss(latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda):
    assert regress_out.shape.as_list()[1] == (D_global_size + 2 * C_global_size)
    # Discrete latents loss
    I_loss_D = 0
    if D_global_size > 0:
        prob_D = tf.nn.softmax(regress_out[:, :D_global_size], axis=1)
        I_loss_D = tf.reduce_sum(latents[:, :D_global_size] * tf.log(prob_D + 1e-12), axis=1)
    # Continuous latents loss
    mean_C = regress_out[:, D_global_size:D_global_size + C_global_size]
    std_C = tf.sqrt(tf.exp(regress_out[:, D_global_size + C_global_size: D_global_size + C_global_size * 2]))
    epsilon = (latents[:, D_global_size:] - mean_C) / (std_C + 1e-12)
    I_loss_C = tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(std_C + 1e-12) - 0.5 * tf.square(epsilon), axis=1)
    I_loss = - D_lambda * I_loss_D - C_lambda * I_loss_C
    return I_loss

def G_logistic_ns_info_gan(G, D, I, opt, training_set, minibatch_size, latent_type='uniform', D_global_size=0, D_lambda=1, C_lambda=1):
    _ = opt
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, _ = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, hidden = D.get_output_for(fake_images_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    # regress_out = I.get_output_for(hidden, is_training=True)
    regress_out = I.get_output_for(fake_images_out, is_training=True)
    I_loss = calc_info_gan_loss(latents, regress_out, D_global_size, G.input_shapes[0][1]-D_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)
    return G_loss, None, I_loss, None

def calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type):
    assert regress_out.shape.as_list()[1] == (D_global_size + C_global_size)
    # Continuous latents loss
    if delta_type == 'onedim':
        prob_C = tf.nn.softmax(regress_out[:, D_global_size:], axis=1)
        I_loss_C = C_delta_latents * tf.log(prob_C + 1e-12)
        I_loss_C = C_lambda * I_loss_C

        I_loss_C = tf.reduce_sum(I_loss_C, axis=1)
        I_loss = - I_loss_C
    elif delta_type == 'fulldim':
        I_loss_C = tf.reduce_sum((tf.nn.sigmoid(regress_out[:, D_global_size:]) - C_delta_latents) ** 2, axis=1)
        I_loss = C_lambda * I_loss_C
    return I_loss

# def calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda):
    # assert regress_out.shape.as_list()[1] == (D_global_size + C_global_size)
    # # Continuous latents loss
    # I_loss_C = tf.reduce_mean((regress_out[:, D_global_size:] - delta_target) ** 2, axis=1)
    # I_loss = C_lambda * I_loss_C
    # return I_loss

def calc_cls_loss(discrete_latents, cls_out, D_global_size, C_global_size, cls_alpha):
    assert cls_out.shape.as_list()[1] == D_global_size
    prob_D = tf.nn.softmax(cls_out, axis=1)
    I_info_loss_D = tf.reduce_sum(discrete_latents * tf.log(prob_D + 1e-12), axis=1)
    I_info_loss = - cls_alpha * I_info_loss_D
    return I_info_loss

def G_logistic_ns_vc(G, D, I, opt, training_set, minibatch_size, I_info=None, latent_type='uniform',
                     D_global_size=0, D_lambda=0, C_lambda=1, F_beta=0, cls_alpha=0, epsilon=0.4,
                     random_eps=False, delta_type='onedim', cascading=False):
    _ = opt
    discrete_latents = None
    C_global_size = G.input_shapes[0][1]-D_global_size
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)
        discrete_latents_2 = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents_2 = tf.one_hot(discrete_latents_2, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        # latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
        latents = tf.random.normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)

    if not cascading:
        # Sample delta latents
        if delta_type == 'onedim':
            C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=C_global_size, dtype=tf.int32)
            C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, C_global_size), latents.dtype)
        elif delta_type == 'fulldim':
            C_delta_latents = tf.random.uniform([minibatch_size, C_global_size], minval=0, maxval=1.0, dtype=latents.dtype)
    else:
        # apply cascading
        cascade_max = 1e5
        cascade_step = cascade_max // int(C_global_size)
        global_step = tf.compat.v1.train.get_global_step()
        n_emph_free = tf.math.floormod(global_step // int(cascade_step), C_global_size) + 2
        n_emph = tf.math.minimum(n_emph_free, C_global_size)

        if delta_type == 'onedim':
            C_delta_latents = tf.random.uniform([minibatch_size], minval=0, maxval=n_emph, dtype=tf.int32)
            C_delta_latents = tf.cast(tf.one_hot(C_delta_latents, n_emph), latents.dtype)
        elif delta_type == 'fulldim':
            C_delta_latents = tf.random.uniform([minibatch_size, n_emph], minval=0, maxval=1.0, dtype=latents.dtype)

        C_delta_latents = tf.concat([C_delta_latents, tf.zeros([minibatch_size, C_global_size - n_emph])], axis=1)

    if delta_type == 'onedim':
        if not random_eps:
            delta_target = C_delta_latents * epsilon
            # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
        else:
            epsilon = epsilon * tf.random.normal([minibatch_size, 1], mean=0.0, stddev=2.0)
            # delta_target = tf.math.abs(C_delta_latents * epsilon)
            delta_target = C_delta_latents * epsilon
            # delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_target], axis=1)
    else:
        delta_target = (C_delta_latents - 0.5) * epsilon

    delta_latents = delta_target + latents

    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)
        # delta_latents = tf.concat([discrete_latents_2, delta_latents], axis=1)
        delta_latents = tf.concat([tf.zeros([minibatch_size, D_global_size]), delta_latents], axis=1)

    labels = training_set.get_random_labels_tf(minibatch_size)
    fake1_out, feat_map1 = G.get_output_for(latents, labels, is_training=True)
    fake2_out, feat_map2 = G.get_output_for(delta_latents, labels, is_training=True)
    if I_info is not None:
        fake_scores_out, hidden = D.get_output_for(fake1_out, labels, is_training=True)
    else:
        fake_scores_out = D.get_output_for(fake1_out, labels, is_training=True)
    G_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    
    regress_out = I.get_output_for(fake1_out, fake2_out, is_training=True)
    I_loss = calc_vc_loss(C_delta_latents, regress_out, D_global_size, C_global_size, D_lambda, C_lambda, delta_type)
    # I_loss = calc_vc_loss(delta_target, regress_out, D_global_size, C_global_size, D_lambda, C_lambda)
    I_loss = autosummary('Loss/I_loss', I_loss)

    F_loss = tf.reduce_mean(feat_map1 * feat_map1, axis=[1, 2, 3])
    F_loss = autosummary('Loss/F_loss', F_loss)

    I_loss += (F_loss * F_beta)

    if I_info is not None:
        cls_out = I_info.get_output_for(hidden, is_training=True)
        I_info_loss = calc_cls_loss(discrete_latents, cls_out, D_global_size, G.input_shapes[0][1]-D_global_size, cls_alpha)
        I_info_loss = autosummary('Loss/I_info_loss', I_info_loss)
    else:
        I_info_loss = None
    return G_loss, None, I_loss, I_info_loss

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r1_dsp(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

    fake_images_out, _ = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r1_info_gan(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, latent_type='uniform', D_global_size=0):
    _ = opt, training_set
    discrete_latents = None
    if D_global_size > 0:
        discrete_latents = tf.random.uniform([minibatch_size], minval=0, maxval=D_global_size, dtype=tf.int32)
        discrete_latents = tf.one_hot(discrete_latents, D_global_size)

    if latent_type == 'uniform':
        latents = tf.random.uniform([minibatch_size] + [G.input_shapes[0][1]-D_global_size], minval=-2, maxval=2)
    elif latent_type == 'normal':
        latents = tf.random_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    elif latent_type == 'trunc_normal':
        latents = tf.random.truncated_normal([minibatch_size] + [G.input_shapes[0][1]-D_global_size])
    else:
        raise ValueError('Latent type not supported: ' + latent_type)
    if D_global_size > 0:
        latents = tf.concat([discrete_latents, latents], axis=1)

    fake_images_out, _ = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, _ = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out, _ = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        fake_grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017

def G_wgan(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -fake_scores_out
    return loss, None

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, wgan_epsilon=0.001):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon
    return loss, None

#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = D.get_output_for(mixed_images_out, labels, is_training=True)
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_images_out])[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        reg = gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss, reg

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg_dsp(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_sum(tf.square(pl_grads), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
