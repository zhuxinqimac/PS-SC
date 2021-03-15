#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: dsprites_data_helper.py
# --- Creation Date: 24-05-2020
# --- Last Modified: Wed 04 Nov 2020 16:05:59 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Dataset helper for dsprites.
"""

import numpy as np
import os
import glob
from training import misc

def str_to_intlist(v):
    v = v[1:-1]
    v_list = [int(i.strip()) for i in v.strip().split(',')]
    return v_list

class DisentangleDataHelper:
    def __init__(self, dataset_dir, factor_sizes):
        self.factor_sizes = factor_sizes
        self.factor_bases = np.prod(self.factor_sizes) // np.cumprod(self.factor_sizes)
        self.num_factors = len(self.factor_bases)
        print('factor_bases:', self.factor_bases)
        assert self.factor_bases[-1] == 1

        self.label_file = glob.glob(os.path.join(dataset_dir, '*.labels'))[0]
        self.np_labels = np.load(self.label_file)

        self.data_file = glob.glob(os.path.join(dataset_dir, '*.data'))[0]
        self.np_data = np.load(self.data_file)
        self.n_data = self.np_data.shape[0]

    def sample_factors(self, num, random_state):
        idxs = random_state.randint(self.n_data, size=num)
        return self.batch_data_indices_to_latents(idxs)

    def sample_observations_from_factors(self, factors, random_state):
        idxs = self.batch_latents_to_data_indices(factors)
        images = self.np_data[idxs]
        images = misc.adjust_dynamic_range(images, [0, 255], [-1., 1.])
        return images

    def sample_observations(self, num, random_state):
        idxs = random_state.randint(self.n_data, size=num)
        images = self.np_data[idxs]
        images = misc.adjust_dynamic_range(images, [0, 255], [-1., 1.])
        return images

    def get_data(self, idx):
        return self.np_data[idx]

    def data_index_to_latent(self, idx):
        cur_latent = []
        tmp_idx = idx
        for dim in range(len(self.factor_sizes)):
            cur_latent.append(tmp_idx // self.factor_bases[dim])
            tmp_idx = tmp_idx % self.factor_bases[dim]
        return np.array(cur_latent)

    def batch_data_indices_to_latents(self, idxs):
        cur_latents = []
        tmp_idxs = idxs
        for dim in range(len(self.factor_sizes)):
            cur_latents.append(tmp_idxs // self.factor_bases[dim])
            tmp_idxs = tmp_idxs % self.factor_bases[dim]
        # cur_latents: [n_factors, n_idxs]
        cur_latents = np.transpose(np.array(cur_latents), [1, 0])
        assert cur_latents.shape == (len(idxs), len(self.factor_sizes))
        return np.array(cur_latents)

    def latent_to_data_index(self, latent):
        idx = np.sum(self.factor_bases * latent)
        return idx

    def batch_latents_to_data_indices(self, latents):
        idxs = np.sum(self.factor_bases * latents, axis=1)
        return idxs

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)


class DspritesDataHelper(DisentangleDataHelper):
    def __init__(self, dataset_dir, use_latents='[0,1,2,3,4]'):
        self.full_factor_sizes = np.array([3, 6, 40, 32, 32])
        self.use_latents = str_to_intlist(use_latents)
        self.factor_sizes = self.full_factor_sizes[self.use_latents]
        DisentangleDataHelper.__init__(self, dataset_dir, self.factor_sizes)


class Shape3DDataHelper(DisentangleDataHelper):
    def __init__(self, dataset_dir, use_latents='[0,1,2,3,4,5]'):
        self.full_factor_sizes = np.array([10, 10, 10, 8, 4, 15])
        self.use_latents = str_to_intlist(use_latents)
        self.factor_sizes = self.full_factor_sizes[self.use_latents]
        DisentangleDataHelper.__init__(self, dataset_dir, self.factor_sizes)
