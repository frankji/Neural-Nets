#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:04:35 2017

@author: Frank
"""

import os
os.chdir('/Users/Frank/Courses/workshop/tensorflow/061517')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')


def mnist_cov(features, labels, mode, params):
    W_conv1 = tf.get_variable('W_conv1', params['wconv1_shape'], dtype=tf.float32)
    b_conv1 = tf.get_variable('b_conv1', params['wconv1_shape'][-1], dtype=tf.float32)
    x_image = tf.reshape(features['x'], [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = tf.get_variable('W_conv2', params['wconv2_shape'], dtype=tf.float32)
    b_conv2 = tf.get_variable('b_conv2', params['wconv2_shape'][-1], dtype=tf.float32)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = tf.get_variable('W_fc1', params['wfc1_shape'], dtype=tf.float32)
    b_fc1= tf.get_variable('b_fc1', params['wfc1_shape'][-1], dtype=tf.float32)
    h_pool2_flat = tf.reshape(h_pool2, [-1, params['wfc1_shape'][0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, params['keep_prob'])
    W_fc2 = tf.get_variable('W_fc2', [params['wfc1_shape'][-1], np.shape(labels)[-1]], dtype=tf.float32)
    b_fc2 = tf.get_variable('b_fc2', [np.shape(labels)[-1]], dtype=tf.float32)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
    optimizer = tf.train.AdamOptimizer(params['Learning_rate'])
    global_step = tf.train.get_global_step()
    myjob = tf.group(optimizer.minimize(cross_entropy), tf.assign_add(global_step, 1))
    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    evaluation = {'accuracy': accuracy}
    return tf.contrib.learn.ModelFnOps(mode = mode, predictions = y_conv, loss = cross_entropy, train_op = myjob, eval_metric_ops = evaluation)
params = {'wconv1_shape': [5, 5, 1, 32], 'wconv2_shape': [5, 5, 32, 64], 'wfc1_shape': [7 * 7 * 64, 1024], 'keep_prob':np.float32(0.5), 'Learning_rate':np.float32(0.0001)}
estimator = tf.contrib.learn.Estimator(model_fn = mnist_cov, model_dir='./mnist2', params = params)
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':mnist.train.images}, mnist.train.labels.astype('float32'), batch_size=100, num_epochs=100)
estimator.fit(input_fn=input_fn, steps=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':mnist.test.images}, mnist.test.labels.astype('float32'), batch_size=100, num_epochs=100)
estimator.evaluate(input_fn=eval_input_fn, steps=100)
y_pred = np.argmax(estimator.predict(x={'x': mnist.test.images}, as_iterable=False), 1)
print('Accuracy: %g'%np.mean(y_pred == np.argmax(mnist.test.labels, 1)))

W_mat = estimator.get_variable_value('W_conv1')
cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['dodgerblue','white', 'crimson'], N=256)
fig, axs = plt.subplots(4,8, figsize=(10,4))
for i in range(32):
    axs = plt.subplot(4,8,i+1)
    axs.matshow(W_mat[:,:,0,i], cmap=cmap)
    plt.xticks([])
    plt.yticks([])
fig.savefig('W_conv1.png')
plt.close(fig)

W_mat = estimator.get_variable_value('W_conv2')
cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['dodgerblue','white', 'crimson'], N=256)
fig, axs = plt.subplots(8,8, figsize=(10,10))
for i in range(64):
    axs = plt.subplot(8,8,i+1)
    axs.matshow(W_mat[:,:,0,i], cmap=cmap)
    plt.xticks([])
    plt.yticks([])
fig.savefig('W_conv2.png')
plt.close(fig)

