import tensorflow as tf
import tensorflow.contrib as tf_contrib

import numpy as np


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

#DEAL modifications###################################

def Softmax_dense_layer(x, units, scope='Softmax_dense_0'):
    with tf.variable_scope(scope):
        W3 = var('W3', [x.get_shape()[1].value, 1000])
        b3 = var('b3', [1000])
        out3 = tf.nn.relu(tf.matmul(x, W3) + b3)
        rate=0.5
        if rate > 0:
            print('Dropout enabled!')
        out3 = tf.layers.dropout(out3, rate=rate)

        W4 = var('W4', [1000, units])
        b4 = var('b4', [units])
        logits = tf.matmul(out3, W4) + b4
        return logits


def DEAL_dense_layer(x, units, scope='DEAL_dense_0'):
    with tf.variable_scope(scope):
        W3 = var('W3', [x.get_shape()[1].value, 1000])
        b3 = var('b3', [1000])
        out3 = tf.nn.relu(tf.matmul(x, W3) + b3)

        print('OUT3')
        print(out3.shape)

        rate=0.5
        if rate > 0:
            print('Dropout enabled!')
        out3 = tf.layers.dropout(out3, rate=rate)

        W4 = var('W4', [1000, units])
        b4 = var('b4', [units])
        logits = tf.matmul(out3, W4) + b4
        return logits, W3, W4

######################################################

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')



        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

##################################################################################
# DEAL Modifications

#### Logit to evidence converters - activation functions (they have to produce non-negative outputs for the uncertaintyuncertainity process)

def relu_evidence(logits):
    return tf.nn.relu(logits)

def exp_evidence(logits):
    return tf.exp(logits / 1000)

def relu6_evidence(logits):
    return tf.nn.relu6(logits)

def softsign_evidence(logits):
    return tf.nn.softsign(logits)#

#### Variable Initializers

def var(name, shape, init=None):

    init = tf.truncated_normal_initializer(stddev=(1 / shape[0]) ** 0.5) if init is None else init
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=init)


#### KL Divergence calculator

def KL(alpha, K):
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

    KL = tf.reduce_sum((alpha - beta) * (tf.digamma(alpha) - tf.digamma(S_alpha)), axis=1, keepdims=True) + \
         tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keepdims=True) + \
         tf.reduce_sum(tf.lgamma(beta), axis=1, keepdims=True) - tf.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
    return KL

##### Loss functions (there are three different one defined in the papaer)

def loss_eq5(p, alpha, K, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    loglikelihood = tf.reduce_sum((p - (alpha / S)) ** 2, axis=1, keepdims=True) + tf.reduce_sum(
      alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg

def loss_eq4(p, alpha, K, global_step, annealing_step):
    loglikelihood = tf.reduce_mean(
      tf.reduce_sum(p * (tf.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.digamma(alpha)), 1,
                    keepdims=True))
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg

def loss_eq3(p, alpha, K, global_step, annealing_step):
    loglikelihood = tf.reduce_mean(
      tf.reduce_sum(p * (tf.log(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.log(alpha)), 1, keepdims=True))
    KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * KL((alpha - 1) * (1 - p) + 1, K)
    return loglikelihood + KL_reg


#### Putting the loss functions together

def deal_loss(logits, label, units, W3, W4, lmb, global_step, annealing_step):

    p = label
    alpha = logits
    K = units

    loss_function = loss_eq3(p, alpha, K, global_step, annealing_step)
    loss = tf.reduce_mean(loss_function)
    l2_loss = (tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)) * lmb
    loss = loss + l2_loss

    match = tf.reshape(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), tf.float32), (-1, 1))
    acc = tf.reduce_mean(match)

    return loss, acc



##################################################################################



