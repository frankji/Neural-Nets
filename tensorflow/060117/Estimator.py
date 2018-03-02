#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:50:51 2017

@author: Frank
"""

import tensorflow as tf
import numpy as np
import scipy.stats


x_train = np.array(np.random.rand(50,10), dtype='float32')
beta0 = np.array([1,2,3,4,5,6,7,0.01,0.01,0.01,8], dtype='float32').reshape((11,1))
noise = np.array(scipy.stats.norm(loc=0, scale=1).rvs((50,1)), dtype='float32')
y_train = np.matmul(np.matrix(x_train), beta0[0:10]) + beta0[10] + noise

def linearmodel(features, labels, mode):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
    regularizer=tf.contrib.layers.l1_regularizer(0.5)
    beta = tf.get_variable('beta', [10, 1], dtype=tf.float32, regularizer=regularizer)
    beta0 = tf.get_variable('beta0', [1], dtype=tf.float32)
    y = tf.matmul(features['x'], beta) + beta0
    loss = tf.reduce_sum(tf.square(y - labels))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(mode = mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=linearmodel, model_dir='./model/')
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_train}, y_train, batch_size=5, num_epochs=500)
estimator.fit(input_fn=input_fn, steps=5000)

