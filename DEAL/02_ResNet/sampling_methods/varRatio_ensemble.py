# Copyright 2020 DeepLearningResearch
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by DeepLearningResearch for the development of DEAL.

"""Variation ratio AL method for Softmax and Deep Ensemble approach.

Samples in batches based on variation ratio scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
from scipy.stats import mode
from sampling_methods.sampling_def import SamplingMethod




class VarRatio_ensemble(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'VarRatio_ensemble'
    self.dropout_iterations = 5



  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with highest uncertainty.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using variation ratio active learner
    """

    with open('./trained_models/All_Dropout_Classes', 'rb') as fp:
      All_Dropout_Classes = pickle.load(fp)

    with open('./trained_models/All_Dropout_Classes_dataset', 'rb') as fp:
      All_Dropout_Classes_dataset = pickle.load(fp)

    if All_Dropout_Classes_dataset == 'mnist_keras':
      X_Pool_Dropout = self.X
    if All_Dropout_Classes_dataset == 'cifar10_keras':
      X_Pool_Dropout = self.X
    if All_Dropout_Classes_dataset == 'svhn':
      X_Pool_Dropout = self.X[:86000]
    if All_Dropout_Classes_dataset == 'medical':
      X_Pool_Dropout = self.X[:1400]


    Variation = np.zeros(shape=(X_Pool_Dropout.shape[0]))


    for t in range(X_Pool_Dropout.shape[0]):
      L = np.array([0])
      for d_iter in range(self.dropout_iterations):
        L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
      Predicted_Class, Mode = mode(L[1:])
      v = np.array([1 - Mode / float(self.dropout_iterations)])
      Variation[t] = v

    a_1d = Variation.flatten()


    x_pool_index = a_1d.argsort()[-a_1d.shape[0]:][::-1]

    rank_ind = x_pool_index


    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N]
    return active_samples

