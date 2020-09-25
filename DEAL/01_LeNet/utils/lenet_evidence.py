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

"""Implements LeNet DEAL model in tensorflow backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import keras
import keras.backend as K

import numpy as np
import tensorflow as tf


class LeNet_Evidence(object):

  def __init__(self, random_state=1, epochs=100, batch_size=32, solver='adam', learning_rate=0.001, lr_decay=0.):
    # params
    self.solver = solver
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.lr_decay = lr_decay
    # data
    self.encode_map = None
    self.decode_map = None
    self.model = None
    self.random_state = random_state
    self.n_classes = None

    self.mnist = 28
    self.mnist_y = 1
    self.cifar = 32
    self.cifar_y = 3
    self.audi = 512
    self.audi_y = 3

    self.pixel = self.mnist
    self.channel = self.mnist_y
  #-----------------------------------------------------

    self.g = None
    self.step = None
    self.X = None
    self.Y = None
    self.annealing_step = None
    self.keep_prob = None
    self.prob = None
    self.acc = None
    self.loss = None
    self.u = None
    self.evidence = None
    self.mean_ev = None
    self.mean_ev_succ = None
    self.mean_ev_fail = None
    self.sess = None
    self.saver = None
    self.active_learning_round = 0

    self.extract_features = None

    self.L_train_acc = []
    self.L_train_ev_s = []
    self.L_train_ev_f = []

    self.L_test_acc = []
    self.L_test_ev_s = []
    self.L_test_ev_f = []

  #### Logit to evidence converters - activation functions (they have to produce non-negative outputs for the uncertaintyuncertainity process)

  def relu_evidence(self, logits):
    return tf.nn.relu(logits)

  def exp_evidence(self, logits):
    return tf.exp(logits / 1000)

  def relu6_evidence(self, logits):
    return tf.nn.relu6(logits)

  def softsign_evidence(self, logits):
    return tf.nn.softsign(logits)

  #### KL Divergence calculator

  def KL(self,alpha, K):
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

    KL = tf.reduce_sum((alpha - beta) * (tf.digamma(alpha) - tf.digamma(S_alpha)), axis=1, keepdims=True) + \
         tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keepdims=True) + \
         tf.reduce_sum(tf.lgamma(beta), axis=1, keepdims=True) - tf.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
    return KL

  ##### Loss functions (there are three different one defined in the papaer)

  def loss_eq5(self, p, alpha, K, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    loglikelihood = tf.reduce_sum((p - (alpha / S)) ** 2, axis=1, keepdims=True) + tf.reduce_sum(
      alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * self.KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg

  def loss_eq4(self, p, alpha, K, global_step, annealing_step):
    loglikelihood = tf.reduce_mean(
      tf.reduce_sum(p * (tf.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.digamma(alpha)), 1,
                    keepdims=True))
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * self.KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg

  def loss_eq3(self, p, alpha, K, global_step, annealing_step):
    loglikelihood = tf.reduce_mean(
      tf.reduce_sum(p * (tf.log(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.log(alpha)), 1, keepdims=True))
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * self.KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg

  def var(self, name, shape, init=None):

    init = tf.truncated_normal_initializer(stddev=(1 / shape[0]) ** 0.5) if init is None else init
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=init)

  def LeNet_EDL(self, K, loss_function, logits2evidence=relu_evidence, lmb=0.005, dims=(28, 28), nch=1):
    g = tf.Graph()
    with g.as_default():
      X = tf.placeholder(shape=[None, np.prod(dims) * nch], dtype=tf.float32)
      Y = tf.placeholder(shape=[None, K], dtype=tf.float32)
      keep_prob = tf.placeholder(dtype=tf.float32)
      global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
      annealing_step = tf.placeholder(dtype=tf.int32)

      W1 = self.var('W1', [5, 5, nch, 20])
      b1 = self.var('b1', [20])
      c1 = tf.nn.conv2d(tf.reshape(X, [-1, *dims, nch]), W1, [1, 1, 1, 1], 'SAME')
      r1 = tf.nn.relu(c1 + b1)
      out1 = tf.nn.max_pool(r1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

      W2 = self.var('W2', [5, 5, 20, 50])
      b2 = self.var('b2', [50])
      c2 = tf.nn.conv2d(out1, W2, [1, 1, 1, 1], 'SAME')
      r2 = tf.nn.relu(c2 + b2)
      out2 = tf.nn.max_pool(r2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

      Xflat = tf.contrib.layers.flatten(out2)


      W3 = self.var('W3', [Xflat.get_shape()[1].value, 500])
      b3 = self.var('b3', [500])
      out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)


      #------------------ exctract features--------------------------
      extract_features = out3
      #--------------------------------------------------------------

      out3 = tf.nn.dropout(out3, keep_prob=keep_prob)

      W4 = self.var('W4', [500, K])
      b4 = self.var('b4', [K])
      logits = tf.matmul(out3, W4) + b4

      evidence = logits2evidence(logits)
      alpha = evidence + 1

      u = K / tf.reduce_sum(alpha, axis=1, keepdims=True)

      prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)

      loss = tf.reduce_mean(loss_function(Y, alpha, K, global_step, annealing_step))
      l2_loss = (tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)) * lmb

      step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss + l2_loss, global_step=global_step)

      match = tf.reshape(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32), (-1, 1))
      acc = tf.reduce_mean(match)

      total_evidence = tf.reduce_sum(evidence, 1, keepdims=True)
      mean_ev = tf.reduce_mean(total_evidence)
      mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence, 1, keepdims=True) * match) / tf.reduce_sum(match + 1e-20)
      mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence, 1, keepdims=True) * (1 - match)) / (
                tf.reduce_sum(tf.abs(1 - match)) + 1e-20)

      saver = tf.train.Saver()

      return g, step, X, Y, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail, saver, extract_features

  def build_model(self):
    g, step, X, Y, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail, saver, extract_features = self.LeNet_EDL(
      10, self.loss_eq3, self.softsign_evidence, dims=(self.pixel, self.pixel), nch=self.channel)
    sess = tf.Session(graph=g)
    with g.as_default():
      sess.run(tf.global_variables_initializer())

    self.g = g
    self.step = step
    self.X = X
    self.Y = Y
    self.annealing_step = annealing_step
    self.keep_prob = keep_prob
    self.prob = prob
    self.acc = acc
    self.loss = loss
    self.u = u
    self.evidence = evidence
    self.mean_ev = mean_ev
    self.mean_ev_succ = mean_ev_succ
    self.mean_ev_fail = mean_ev_fail
    self.sess = sess
    self.saver = saver

    self.extract_features = extract_features


  def fit(self, cifar10_x_train, cifar10_y_train):

    self.build_model()

    cifar10_x_train = cifar10_x_train.reshape(cifar10_y_train.shape[0], self.pixel * self.pixel * self.channel)
    cifar10_y_train = self.create_y_mat(cifar10_y_train)
    print('CIFAR')
    print(cifar10_x_train.shape)
    print(cifar10_y_train.shape)

    g = self.g
    step = self.step
    X = self.X
    Y = self.Y
    annealing_step = self.annealing_step
    keep_prob = self.keep_prob
    prob = self.prob
    acc = self.acc
    loss = self.loss
    u = self.u
    evidence = self.evidence
    mean_ev = self.mean_ev
    mean_ev_succ = self.mean_ev_succ
    mean_ev_fail = self.mean_ev_fail
    sess = self.sess


    self.active_learning_round += 1
    epoch = self.epochs
    bsize = 20
    n_batches = cifar10_x_train.shape[0] // bsize
    for e in range(epoch):
      train_acc = 0.
      for i in range(n_batches):
        data = cifar10_x_train[i * bsize:min((i + 1) * bsize, cifar10_x_train.shape[0]), :]
        label = cifar10_y_train[i * bsize:min((i + 1) * bsize, cifar10_y_train.shape[0]), :]
        sess.run(step, feed_dict={X: data, Y: label, keep_prob: .7, annealing_step: 50 * n_batches})
        accur = sess.run(acc, feed_dict={X: data, Y: label, keep_prob: 1.})
        train_acc += accur
        print('epoch %d - %d%%) ' % (e + 1, (100 * (i + 1)) // n_batches), end='\r' if i < n_batches - 1 else '')


      acc_cache = (train_acc / n_batches)

      print('training: %2.4f' %
            (train_acc / n_batches,))
      if acc_cache > (train_acc / n_batches):
        break

      #self.saver.save(sess, './trained_models/lenet_evidence_uncertaintyEDL' + str(self.active_learning_round))


  def create_y_mat(self, y):

    y_mat = np.eye(10)[y]
    return y_mat

  # Add handling for classes that do not start counting from 0
  def encode_y(self, y):
    if self.encode_map is None:
      self.classes_ = sorted(list(set(y)))
      self.n_classes = len(self.classes_)
      self.encode_map = dict(zip(self.classes_, range(len(self.classes_))))
      self.decode_map = dict(zip(range(len(self.classes_)), self.classes_))
    mapper = lambda x: self.encode_map[x]
    transformed_y = np.array(map(mapper, y))
    return transformed_y

  def decode_y(self, y):
    mapper = lambda x: self.decode_map[x]
    transformed_y = np.array(map(mapper, y))
    return transformed_y



  def predict(self, X_val, test_time_dropout):


    X_val = X_val.reshape(X_val.shape[0], self.pixel * self.pixel * self.channel)


    start = 0
    end = 1000
    unc = np.expand_dims(np.array([]), axis=1)
    pred = np.zeros(shape=(0,10))
    features = np.zeros(shape=(0,500))
    for i in range(0,int(X_val.shape[0]/1000)):

      X_cache = X_val[start:end]


      feed_dict = {self.X: X_cache, self.keep_prob: 1.0} # X_val
      p_pred_t, u, feat = self.sess.run([self.prob, self.u, self.extract_features], feed_dict=feed_dict)
      p_pred_t = np.array(p_pred_t)
      u = np.array(u)
      feat = np.array(feat)


      unc = np.concatenate([unc, u])
      pred = np.concatenate([pred, p_pred_t])
      features = np.concatenate([features, feat])


      start += 1000
      end += 1000

    print('Uncertainty Shape')
    print(u.shape)
    print(pred.shape)

    return unc, pred, features

  def score(self, x_test, val_y):
    x_test = x_test.reshape(x_test.shape[0], self.pixel * self.pixel * self.channel)
    y_test = self.create_y_mat(val_y)
    test_acc, test_succ, test_fail = self.sess.run([self.acc,self.mean_ev_succ,self.mean_ev_fail], feed_dict={self.X:x_test,self.Y:y_test,self.keep_prob:1.})
    self.L_test_acc.append((test_acc))
    self.L_test_ev_s.append(test_succ)
    self.L_test_ev_f.append((test_fail))
    val_acc = test_acc

    return val_acc

  def decision_function(self, X, test_time_dropout):
    return self.predict(X, test_time_dropout)

  def transform(self, X):
    model = self.model
    inp = [model.input]
    activations = []

    # Get activations of the last conv layer.
    output = [layer.output for layer in model.layers if
              layer.name == 'conv9'][0]
    func = K.function(inp + [K.learning_phase()], [output])
    for i in range(int(X.shape[0]/self.batch_size) + 1):
      minibatch = X[i * self.batch_size
                    : min(X.shape[0], (i+1) * self.batch_size)]
      list_inputs = [minibatch, 0.]
      # Learning phase. 0 = Test mode (no dropout or batch normalization)
      layer_output = func(list_inputs)[0]
      activations.append(layer_output)
    output = np.vstack(tuple(activations))
    output = np.reshape(output, (output.shape[0],np.product(output.shape[1:])))
    return output

  def get_params(self, deep = False):
    params = {}
    params['solver'] = self.solver
    params['epochs'] = self.epochs
    params['batch_size'] = self.batch_size
    params['learning_rate'] = self.learning_rate
    params['weight_decay'] = self.lr_decay
    if deep:
      return copy.deepcopy(params)
    return copy.copy(params)

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self
