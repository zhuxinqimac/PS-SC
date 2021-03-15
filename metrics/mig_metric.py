#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: mig_metric.py
# --- Creation Date: 04-11-2020
# --- Last Modified: Wed 04 Nov 2020 17:13:10 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""MIG metric."""

import numpy as np
import pdb
import tensorflow as tf
import scipy
import sklearn
from sklearn import ensemble
import dnnlib.tflib as tflib

from metrics import metric_base
from metrics.perceptual_path_length import normalize, slerp
from training import misc
from disentanglement_data_helper import DspritesDataHelper, Shape3DDataHelper
from training.utils import get_return_v

#----------------------------------------------------------------------------


class MIGMetric(metric_base.MetricBase):
    def __init__(self,
                 dataset_dir,
                 dataset_name,
                 use_latents,
                 batch_size,
                 num_train,
                 has_label_place=False,
                 drange_net=[-1., 1.],
                 **kwargs):
        super().__init__(**kwargs)
        if dataset_name == 'Dsprites':
            self.ground_truth_data = DspritesDataHelper(
                dataset_dir, use_latents)
        elif dataset_name == '3DShapes':
            self.ground_truth_data = Shape3DDataHelper(dataset_dir,
                                                       use_latents)
        else:
            raise ValueError('Not recognized dataset:', dataset_name)
        self.batch_size = batch_size
        self.num_train = num_train
        self.has_label_place = has_label_place
        self.drange_net = drange_net

    def _evaluate(self, I_net, **kwargs):
        representation_model = I_net.clone(is_validation=True)
        random_state = np.random.RandomState(123)

        mus_train, ys_train = self.generate_batch_factor_code(
            self.ground_truth_data, representation_model, self.num_train,
            random_state, self.batch_size)
        assert mus_train.shape[1] == self.num_train
        scores = self._compute_mig(mus_train, ys_train)
        self._report_result(scores["discrete_mig"], suffix='_discrete_mig')
        print('scores_dict:', scores)

    def _compute_mig(self, mus_train, ys_train):
        """Computes score based on both training and testing codes and factors."""
        score_dict = {}
        # discretized_mus = utils.make_discretizer(mus_train)
        discretized_mus = self._histogram_discretize(mus_train)
        m = self.discrete_mutual_info(discretized_mus, ys_train)
        assert m.shape[0] == mus_train.shape[0]
        assert m.shape[1] == ys_train.shape[0]
        # m is [num_latents, num_factors]
        entropy = self.discrete_entropy(ys_train)
        sorted_m = np.sort(m, axis=0)[::-1]
        score_dict["discrete_mig"] = np.mean(
            np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        return score_dict

    def _histogram_discretize(self, target, num_bins=20):
        """Discretization based on histograms."""
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(target[i, :], np.histogram(
                target[i, :], num_bins)[1][:-1])
        return discretized

    def discrete_entropy(self, ys):
        """Compute discrete mutual information."""
        num_factors = ys.shape[0]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
        return h

    def discrete_mutual_info(self, mus, ys):
        """Compute discrete mutual information."""
        num_codes = mus.shape[0]
        num_factors = ys.shape[0]
        m = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):
                m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
        return m

    def generate_batch_factor_code(self, ground_truth_data,
                                   representation_function, num_points,
                                   random_state, batch_size):
        """Sample a single training sample based on a mini-batch of ground-truth data.

        Args:
            ground_truth_data: GroundTruthData to be sampled from.
            representation_function: Function that takes observation as input and
              outputs a representation.
            num_points: Number of points to sample.
            random_state: Numpy random state used for randomness.
            batch_size: Batchsize to sample points.

        Returns:
            representations: Codes (num_codes, num_points)-np array.
            factors: Factors generating the codes (num_factors, num_points)-np array.
        """
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_factors, current_observations = \
                ground_truth_data.sample(num_points_iter, random_state)
            current_observations = misc.adjust_dynamic_range(current_observations, [-1., 1.], self.drange_net)
            if i == 0:
                factors = current_factors
                # representations = representation_function(current_observations)
                if self.has_label_place:
                    representations = get_return_v(
                        representation_function.run(
                            current_observations,
                            np.zeros([current_observations.shape[0], 0]),
                            is_validation=True), 1)
                else:
                    representations = get_return_v(
                        representation_function.run(current_observations,
                                                    is_validation=True), 1)
            else:
                factors = np.vstack((factors, current_factors))
                if self.has_label_place:
                    representations_i = get_return_v(
                        representation_function.run(
                            current_observations,
                            np.zeros([current_observations.shape[0], 0]),
                            is_validation=True), 1)
                else:
                    representations_i = get_return_v(
                        representation_function.run(current_observations,
                                                    is_validation=True), 1)
                # representations = np.vstack((representations,
                # representation_function(
                # current_observations)))
                representations = np.vstack(
                    (representations, representations_i))
            i += num_points_iter
        return np.transpose(representations), np.transpose(factors)
