#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:08:48 2017

@author: Frank
"""

import os
os.chdir('/Users/Frank/Courses/workshop/tensorflow/060117/')
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt 
def sigmoid(Z):
    y = 1/(1+np.exp(-Z))
    return y

def logis(beta, X):
    X = np.matrix(X)
    Z = np.matmul(X, beta)
    return sigmoid(Z)

x = np.concatenate((scipy.stats.norm(loc = -0.1).rvs((500, 10)), scipy.stats.norm(loc = 0.1).rvs((500, 10))))
x = np.concatenate((np.ones((1000,1)),x), 1)
beta0 = scipy.stats.norm(loc=0, scale=1).rvs((11,1))
Z = np.matmul(x, beta0) + scipy.stats.norm(loc = 0, scale=0.1).rvs((1000,1))
yp = 1/(1+np.exp(Z))
y = scipy.stats.bernoulli(p=yp).rvs(yp.shape)
plt.hist(yp)

import tensorflow as tf
beta1 = scipy.stats.norm(loc=0, scale=1).rvs((11,1))
beta = tf.Variable(beta1, dtype=tf.float64)
X = tf.placeholder(dtype=tf.float64)
Y = tf.placeholder(dtype=tf.float64)
h = tf.sigmoid(-tf.matmul(X, beta))
loss = -tf.reduce_sum(y*tf.log(h)+(1-y)*tf.log(1-h))

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(0.005)
myjob = optimizer.minimize(loss)
sess = tf.Session()
sess.run(init)
for i in range(5000):
    sess.run(myjob, {X: x, Y:y})
