#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
File Name: MNIST.py
Author: Jianhao
Mail: daipku@163.com
Created Time: 2018-07-19
"""

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np
import tensorflow as tf

def pic_save(dataSet):
    
    save_dir = '../../data/mnist_data/raw/'

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    for i in range(20):
        image_array = dataSet.train.images[i, :]
        image_array = image_array.reshape(28, 28)

        filename = save_dir + 'mnist_data_%d.jpg' % i
        scipy.misc.toimage(image_array, cmin = 0.0, cmax = 1.0).save(filename)

def label(dataSet):

    for i in range(20):
        one_hot_label = dataSet.train.labels[i, :]
        label = np.argmax(one_hot_label)

        print("mnist_train_%d.jpg label: %d" % (i, label))
    
def softmax_regression(dataSet):

    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        
        batch_xs, batch_ys = dataSet.train.next_batch(100)
        sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict = {x: dataSet.test.images, y_: dataSet.test.labels}))

def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2,1], padding = 'SAME')

def convolutional(dataSet):
    
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = dataSet.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuray %g" % accuracy.eval(feed_dict = {x: dataSet.test.images, y_: dataSet.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':

    mnist = input_data.read_data_sets('../../data/mnist_data/', one_hot = True)
    
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    print(mnist.validation.images.shape)
    print(mnist.validation.labels.shape)

    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)

    print(mnist.train.images[0, :])

    pic_save(mnist)

    print(mnist.train.labels[0, :])

    label(mnist)

    # softmax regression
    softmax_regression(mnist)

	# convolutional
    convolutional(mnist)
