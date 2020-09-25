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

"""Entropy based AL method for DEAL models.

Samples in batches based on entropy scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
from sampling_methods.sampling_def import SamplingMethod


class entropyEDL(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'entropyEDL'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with highest uncertainty according to entropy.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using entropy active learner
    """

    try:
      distances, prediction, features = model.decision_function(self.X)
    except:
      distances, prediction, features = model.predict_proba(self.X)


    entropies_edl1 = [scipy.stats.entropy(prediction[i, :]) for i in range(prediction.shape[0])]
    entropies_edl1 = np.array(entropies_edl1)

    sort_entropies = -np.sort(-entropies_edl1, axis=None)
    print('SORT Entropies')
    print(sort_entropies)
    rank_ind_entropies = np.argsort(-entropies_edl1, axis=None)
    print('RANK IND')
    print(rank_ind_entropies)
    print(entropies_edl1[rank_ind_entropies[0]])


    rank_ind = rank_ind_entropies


    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N]
    return active_samples

