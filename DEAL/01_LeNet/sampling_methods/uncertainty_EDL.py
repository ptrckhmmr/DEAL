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

"""Uncertainty based AL method for DEAL.

Samples in batches based on u scores (see Paper for the exact Equation).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod


class uncertaintyEDL(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'uncertaintyEDL'

    self.test_time_dropout = False

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest highest uncertainty u .

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using uncertainty u active learner
    """

    try:
      uncertainty, prediction, features = model.decision_function(self.X, self.test_time_dropout)
    except:
      uncertainty, prediction, features = model.predict_proba(self.X, self.test_time_dropout)


    distances = uncertainty
    distances = np.array(distances)


    #Sort Array in descending order
    sort_distances = -np.sort(-distances, axis=None)
    print('SORT DISTANCES')
    print(sort_distances)


    rank_ind = np.argsort(-distances, axis=None)

    print('RANK IND')
    print(rank_ind)
    print('Highest Uncertainty')
    print(distances[rank_ind[0]])
    print(prediction[rank_ind[0]])
    print(('Least Uncertainty'))
    print(distances[rank_ind[-1]])
    print(prediction[rank_ind[-1]])


    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N]
    return active_samples

