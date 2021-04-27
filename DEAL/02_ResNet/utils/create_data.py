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

"""Make datasets and save specified directory.

Downloads datasets using scikit datasets and can also parse csv file
to save into pickle format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os
import pickle

try:
   from StringIO import StringIO
except ImportError:
   from io import StringIO


import tarfile
from urllib.request import urlopen

import keras.backend as K
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist

import numpy as np
import pandas as pd


import sklearn.datasets.rcv1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from absl import app
from absl import flags
from tensorflow import gfile

# Flags to specify save directory and data set to be downloaded.


flags.DEFINE_string('save_dir', '../../data/',   # './tmp/data'
                    'Where to save outputs')
flags.DEFINE_string('datasets', 'cifar10_keras',
                    'Which datasets to download, comma separated.')
FLAGS = flags.FLAGS


class Dataset(object):

  def __init__(self, X, y):
    self.data = X
    self.target = y



def get_keras_data(dataname):
  """Get datasets using keras API and return as a Dataset object."""
  if dataname == 'cifar10_keras':
    train, test = cifar10.load_data()
  elif dataname == 'cifar100_coarse_keras':
    train, test = cifar100.load_data('coarse')
  elif dataname == 'cifar100_keras':
    train, test = cifar100.load_data()
  elif dataname == 'mnist_keras':
    train, test = mnist.load_data()
  else:
    raise NotImplementedError('dataset not supported')

  X = np.concatenate((train[0], test[0]))
  y = np.concatenate((train[1], test[1]))

  if dataname == 'mnist_keras':
    # Add extra dimension for channel
    num_rows = X.shape[1]
    num_cols = X.shape[2]
    X = X.reshape(X.shape[0], 1, num_rows, num_cols)
    if K.image_data_format() == 'channels_last':
      X = X.transpose(0, 2, 3, 1)

  y = y.flatten()
  data = Dataset(X, y)
  return data


def get_cifar10():
  """Get CIFAR-10 dataset from source dir.

  Slightly redundant with keras function to get cifar10 but this returns
  in flat format instead of keras numpy image tensor.
  """
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  def download_file(url):
    #req = urllib2.Request(url)
    #response = urllib2.urlopen(req)
    response = urlopen(url).read()
    return response
  response = download_file(url)
  tmpfile = BytesIO()
  while True:
    # Download a piece of the file from the connection
    s = response.read(16384)
    # Once the entire file has been downloaded, tarfile returns b''
    # (the empty bytes) which is a falsey value
    if not s:
      break
    # Otherwise, write the piece of the file to the temporary file.
    tmpfile.write(s)
  response.close()

  tmpfile.seek(0)
  tar_dir = tarfile.open(mode='r:gz', fileobj=tmpfile)
  X = None
  y = None
  for member in tar_dir.getnames():
    if '_batch' in member:
      filestream = tar_dir.extractfile(member).read()
      batch = pickle.load(StringIO.StringIO(filestream))
      if X is None:
        X = np.array(batch['data'], dtype=np.uint8)
        y = np.array(batch['labels'])
      else:
        X = np.concatenate((X, np.array(batch['data'], dtype=np.uint8)))
        y = np.concatenate((y, np.array(batch['labels'])))
  data = Dataset(X, y)
  return data


def get_mldata(dataset):
  # Use scikit to grab datasets and save them save_dir.
  save_dir = FLAGS.save_dir
  filename = os.path.join(save_dir, dataset[1]+'.pkl')

  print('FILENAME')
  print(filename)


  if dataset[0] == 'cifar10_keras':
    data = get_keras_data(dataset[0])
  elif dataset[0] == 'mnist_keras':
    data = get_keras_data(dataset[0])
  elif dataset[0] == 'cifar100_keras':
    data = get_keras_data((dataset[0]))

  X = data.data
  y = data.target
  print(X.shape)
  print(y.shape)
  if X.shape[0] != y.shape[0]:
    X = np.transpose(X)
  assert X.shape[0] == y.shape[0]

  data = {'data': X, 'target': y}

  pickle.dump(data, gfile.GFile(filename, 'w'))


def main(argv):
  del argv  # Unused.

  datasets =  [('cifar10_keras', 'cifar10_keras'), ('mnist_keras', 'mnist_keras'), ('cifar100_keras', 'cifar100_keras')]


  if FLAGS.datasets:
    subset = FLAGS.datasets.split(',')
    datasets = [d for d in datasets if d[1] in subset]

  for d in datasets:
    print(d[1])
    print('D0:', d[0])
    get_mldata(d)


if __name__ == '__main__':
  app.run(main)
