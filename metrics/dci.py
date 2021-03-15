#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: dci.py
# --- Creation Date: 04-11-2020
# --- Last Modified: Wed 04 Nov 2020 16:59:10 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""DCI metric."""

import numpy as np
import pdb
import tensorflow as tf
import scipy
from sklearn import ensemble
import dnnlib.tflib as tflib

from metrics import metric_base
from metrics.perceptual_path_length import normalize, slerp
from training import misc
from disentanglement_data_helper import DspritesDataHelper, Shape3DDataHelper
from training.utils import get_return_v

#----------------------------------------------------------------------------


class DCIMetric(metric_base.MetricBase):
    def __init__(self,
                 dataset_dir,
                 dataset_name,
                 use_latents,
                 batch_size,
                 num_train,
                 num_eval,
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
        self.num_eval = num_eval
        self.has_label_place = has_label_place
        self.drange_net = drange_net

    def _evaluate(self, I_net, **kwargs):
        representation_model = I_net.clone(is_validation=True)
        random_state = np.random.RandomState(123)
        # mus_train are of shape [num_codes, num_train], while ys_train are of shape
        # [num_factors, num_train].
        mus_train, ys_train = self.generate_batch_factor_code(
            self.ground_truth_data, representation_model, self.num_train,
            random_state, self.batch_size)
        assert mus_train.shape[1] == self.num_train
        assert ys_train.shape[1] == self.num_train
        mus_test, ys_test = self.generate_batch_factor_code(
            self.ground_truth_data, representation_model, self.num_eval,
            random_state, self.batch_size)
        scores = self._compute_dci(mus_train, ys_train, mus_test, ys_test)
        self._report_result(scores["informativeness_train"], suffix='_train_inform')
        self._report_result(scores["informativeness_test"], suffix='_test_inform')
        self._report_result(scores["disentanglement"], suffix='_disentanglement')
        self._report_result(scores["completeness"], suffix='_completeness')
        print('scores_dict:', scores)

    def _compute_dci(self, mus_train, ys_train, mus_test, ys_test):
        """Computes score based on both training and testing codes and factors."""
        scores = {}
        importance_matrix, train_err, test_err = self.compute_importance_gbt(
            mus_train, ys_train, mus_test, ys_test)
        assert importance_matrix.shape[0] == mus_train.shape[0]
        assert importance_matrix.shape[1] == ys_train.shape[0]
        scores["informativeness_train"] = train_err
        scores["informativeness_test"] = test_err
        scores["disentanglement"] = self.disentanglement(importance_matrix)
        scores["completeness"] = self.completeness(importance_matrix)
        return scores

    def compute_importance_gbt(self, x_train, y_train, x_test, y_test):
        """Compute importance based on gradient boosted trees."""
        num_factors = y_train.shape[0]
        num_codes = x_train.shape[0]
        importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                     dtype=np.float64)
        train_loss = []
        test_loss = []
        for i in range(num_factors):
            model = ensemble.GradientBoostingClassifier()
            model.fit(x_train.T, y_train[i, :])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(
                np.mean(model.predict(x_train.T) == y_train[i, :]))
            test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
        return importance_matrix, np.mean(train_loss), np.mean(test_loss)

    def disentanglement_per_code(self, importance_matrix):
        """Compute disentanglement score of each code."""
        # importance_matrix is of shape [num_codes, num_factors].
        return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                        base=importance_matrix.shape[1])

    def disentanglement(self, importance_matrix):
        """Compute the disentanglement score of the representation."""
        per_code = self.disentanglement_per_code(importance_matrix)
        if importance_matrix.sum() == 0.:
            importance_matrix = np.ones_like(importance_matrix)
        code_importance = importance_matrix.sum(
            axis=1) / importance_matrix.sum()

        return np.sum(per_code * code_importance)

    def completeness_per_factor(self, importance_matrix):
        """Compute completeness of each factor."""
        # importance_matrix is of shape [num_codes, num_factors].
        return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                        base=importance_matrix.shape[0])

    def completeness(self, importance_matrix):
        """"Compute completeness of the representation."""
        per_factor = self.completeness_per_factor(importance_matrix)
        if importance_matrix.sum() == 0.:
            importance_matrix = np.ones_like(importance_matrix)
        factor_importance = importance_matrix.sum(
            axis=0) / importance_matrix.sum()
        return np.sum(per_factor * factor_importance)

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
