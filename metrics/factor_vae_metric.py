#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: factor_vae_metric.py
# --- Creation Date: 24-05-2020
# --- Last Modified: Mon 02 Nov 2020 13:25:01 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""FactorVAE metric."""

import numpy as np
import pdb
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from metrics.perceptual_path_length import normalize, slerp
from training import misc
from disentanglement_data_helper import DspritesDataHelper, Shape3DDataHelper
from training.utils import get_return_v

#----------------------------------------------------------------------------

class FactorVAEMetric(metric_base.MetricBase):
    def __init__(self, dataset_dir, dataset_name, use_latents, batch_size, num_train,
                 num_eval, num_variance_estimate, has_label_place=False,
                 drange_net=[-1., 1.], **kwargs):
        super().__init__(**kwargs)
        if dataset_name == 'Dsprites':
            self.ground_truth_data = DspritesDataHelper(dataset_dir, use_latents)
        elif dataset_name == '3DShapes':
            self.ground_truth_data = Shape3DDataHelper(dataset_dir, use_latents)
        else:
            raise ValueError('Not recognized dataset:', dataset_name)
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_eval = num_eval
        self.num_variance_estimate = num_variance_estimate
        self.has_label_place = has_label_place
        self.drange_net = drange_net

    def _evaluate(self, I_net, **kwargs):
        representation_model = I_net.clone(is_validation=True)
        random_state = np.random.RandomState(123)
        global_variances = self._compute_variances(representation_model,
                                                   self.num_variance_estimate, random_state)
        active_dims = self._prune_dims(global_variances)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["train_accuracy"] = 0.
            scores_dict["eval_accuracy"] = 0.
            scores_dict["num_active_dims"] = 0
            self._report_result(0., suffix='_train_acc')
            self._report_result(0., suffix='_eval_acc')
            self._report_result(0., suffix='_act_dim')
            return scores_dict

        print("Generating training set.")
        training_votes = self._generate_training_batch(representation_model, self.batch_size,
                                                       self.num_train, random_state,
                                                       global_variances, active_dims)
        print('training_votes:', training_votes)
        classifier = np.argmax(training_votes, axis=0)
        other_index = np.arange(training_votes.shape[1])

        print("Evaluate training set accuracy.")
        train_accuracy = np.sum(
            training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
        print("Training set accuracy: %.2g", train_accuracy)

        print("Generating evaluation set.")
        eval_votes = self._generate_training_batch(representation_model, self.batch_size,
                                              self.num_eval, random_state,
                                              global_variances, active_dims)

        print('eval votes:', eval_votes)
        print("Evaluate evaluation set accuracy.")
        eval_accuracy = np.sum(eval_votes[classifier,
                                          other_index]) * 1. / np.sum(eval_votes)
        print("Evaluation set accuracy: %.2g", eval_accuracy)
        # def _report_result(self, value, suffix='', fmt='%-10.4f'):
        scores_dict["train_accuracy"] = train_accuracy
        scores_dict["eval_accuracy"] = eval_accuracy
        scores_dict["num_active_dims"] = np.sum(active_dims.astype(int))
        self._report_result(train_accuracy, suffix='_train_acc')
        self._report_result(eval_accuracy, suffix='_eval_acc')
        self._report_result(np.sum(active_dims.astype(int)), suffix='_act_dim')
        # return scores_dict
        print('scores_dict:', scores_dict)

    def _prune_dims(self, variances, threshold=0.1):
        scale_z = np.sqrt(variances)
        print('scale_z:', scale_z)
        return scale_z >= threshold

    def _compute_variances(self, representation_model,
                           batch_size,
                           random_state,
                           eval_batch_size=50):
        representations_ls = []
        for i in range(batch_size // 100):
            observations = self.ground_truth_data.sample_observations(100, random_state)
            observations = misc.adjust_dynamic_range(observations, [-1., 1.], self.drange_net)
            # representations = utils.obtain_representation(observations,
                                                          # representation_model,
                                                          # eval_batch_size)
            if self.has_label_place:
                representations = get_return_v(representation_model.run(observations,
                                                                        np.zeros([observations.shape[0], 0]),
                                                                        is_validation=True), 1)
            else:
                representations = get_return_v(representation_model.run(observations, is_validation=True), 1)
            representations_ls.append(representations)
        representations = np.concatenate(tuple(representations_ls), axis=0)
        # representations = np.transpose(representations)
        assert representations.shape[0] == batch_size
        return np.var(representations, axis=0, ddof=1)

    def _generate_training_batch(self, representation_model,
                                 batch_size, num_points, random_state,
                                 global_variances, active_dims):
        votes = np.zeros((self.ground_truth_data.num_factors, global_variances.shape[0]),
                         dtype=np.int64)
        for _ in range(num_points):
            factor_index, argmin = self._generate_training_sample(representation_model,
                                                             batch_size, random_state,
                                                             global_variances,
                                                             active_dims)
            votes[factor_index, argmin] += 1
        return votes

    def _generate_training_sample(self, representation_model,
                                  batch_size, random_state, global_variances,
                                  active_dims):
        # Select random coordinate to keep fixed.
        factor_index = random_state.randint(self.ground_truth_data.num_factors)
        # Sample two mini batches of latent variables.
        factors = self.ground_truth_data.sample_factors(batch_size, random_state)
        # Fix the selected factor across mini-batch.
        factors[:, factor_index] = factors[0, factor_index]
        # Obtain the observations.
        observations = self.ground_truth_data.sample_observations_from_factors(
            factors, random_state)
        observations = misc.adjust_dynamic_range(observations, [-1., 1.], self.drange_net)
        # pdb.set_trace()
        if self.has_label_place:
            representations = get_return_v(representation_model.run(observations,
                                                                    np.zeros([observations.shape[0], 0]),
                                                                    is_validation=True), 1)
        else:
            representations = get_return_v(representation_model.run(observations, is_validation=True), 1)
        local_variances = np.var(representations, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] /
                           global_variances[active_dims])
        return factor_index, argmin
