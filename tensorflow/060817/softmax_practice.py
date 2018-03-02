#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:05:51 2017

@author: Frank
"""

import os
os.chdir('/Users/Frank/Courses/workshop/tensorflow/060817')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST", one_hot = True)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def softmax(X, W, b):
    exps = tf.exp(tf.matmul(X, W) + b)
    partition = tf.reduce_sum(exps, axis = 1)
    function_to_map = lambda x: tf.divide(x, partition)
    y = tf.transpose(tf.map_fn(function_to_map, tf.transpose(exps)))
    return y


def softmax_reg(features, labels, mode, params):
    W = tf.get_variable('W', [784, 10], dtype=tf.float32)
    b = tf.get_variable('b', [10], dtype=tf.float32)
    y = softmax(features['x'], W, b)
    loss = tf.reduce_sum(-tf.reduce_sum(labels*tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(params['Learning_rate'])
    global_step = tf.train.get_global_step()
    myjob = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    evaluation = {'accuracy': accuracy}
    return tf.contrib.learn.ModelFnOps(mode = mode, predictions = y, loss = loss, train_op = myjob, eval_metric_ops = evaluation)

estimator = tf.contrib.learn.Estimator(model_fn = softmax_reg, model_dir='./softmax', params = {'Learning_rate': np.float32(0.02)})
input_fn = tf.contrib.learn.io.numpy_input_fn({'x': mnist.train.images}, mnist.train.labels.astype('float32'), batch_size=100, num_epochs=500)
estimator.fit(input_fn=input_fn, steps=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':mnist.test.images}, mnist.test.labels.astype('float32'), batch_size=100, num_epochs=500)
estimator.evaluate(input_fn=eval_input_fn, steps=1000)
y_pred = np.argmax(estimator.predict(x={'x': mnist.test.images}, as_iterable=False), 1)
