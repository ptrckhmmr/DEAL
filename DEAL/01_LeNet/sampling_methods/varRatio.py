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

"""Variation ratio AL method for Softmax.

Samples in batches based on variation ratio scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import mode
from sampling_methods.sampling_def import SamplingMethod


class VarRatio(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'VarRatio'

    self.dropout_iterations = 25

    self.test_time_dropout = True

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with highest uncertainty.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using variation ratio active learner
    """

    X_Pool_Dropout = self.X


    All_Dropout_Classes = np.zeros(shape=(X_Pool_Dropout.shape[0], 1))
    print('Use trained model for test time dropout')


    for d in range(self.dropout_iterations):
      print('Dropout Iteration', d)


      try:
        pred = model.decision_function(self.X, self.test_time_dropout)
      except:
        pred = model.predict_proba(self.X)


      dropout_classes = np.argmax(pred, axis=1)
      print(dropout_classes.shape)
      dropout_classes = np.array([dropout_classes]).T

      All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

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

