#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:10:45 2017

@author: Frank
"""

import tensorflow as tf
import numpy as np
import scipy.stats
features = [tf.contrib.layers.real_valued_column('x', dimension=4)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features, model_dir='./model/')
x_train = np.array(np.random.rand(1000,4), dtype='float32')
### Pretend we dont know it ###
beta0 = np.array([1,2,3,4,5], dtype='float32').reshape((5,1))
noise = np.array(scipy.stats.norm(loc=0, scale=1).rvs((1000,1)), dtype='float32')
y_train = np.matmul(np.matrix(x_train), beta0[0:4]) + beta0[4] + noise
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_train}, y_train, batch_size=50, num_epochs=200)

estimator.fit(input_fn=input_fn, steps = 10000)
print(estimator.evaluate(input_fn=input_fn))
estimator.get_variable_names()
estimator.get_variable_value('linear/bias_weight')
estimator.get_variable_value('linear/x/weight')

x_eval = np.array(np.random.rand(1000,4), dtype='float32')
y_eval = np.matmul(np.matrix(x_eval), beta0[0:4]) + beta0[4]
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_eval}, y_eval, batch_size=50, num_epochs=200)

train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
y_pred = np.array([i for i in estimator.predict({'x':x_eval})])


