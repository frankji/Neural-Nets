#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:40:52 2017

@author: Frank
"""
import os
os.chdir('/Users/Frank/Courses/workshop/tensorflow/060817')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST", one_hot = True)
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
#yprime = tf.placeholder(tf.float32)
#y_ = tf.one_hot(tf.cast(yprime, tf.int32), 10)
cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x:batch_xs, y_: batch_ys})
    print(sess.run([accuracy], feed_dict={x:batch_xs, y_: batch_ys}))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
