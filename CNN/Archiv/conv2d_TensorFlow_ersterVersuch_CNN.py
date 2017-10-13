#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:54:12 2017

@author: Chris

A Convolutional Network implementation using TensorFlow library.
"""

from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np

# import data into numpy arrays
data = pd.read_csv('/Users/Chris/Python/Machine Learning/Masterarbeit1/ExpPosExpo_maturity_fixedCoupon_seed-JavaExport.csv', header=None, sep=";")
X_data_pure = data.iloc[:, : data.shape[1]-2 ].values
X_lables_pure = data.iloc[:, data.shape[1]-2 : data.shape[1] - 1 ].values

# clean the label array
a = np.zeros((2250,15)) 
for i, number in enumerate(X_lables_pure):
    a[i][ int(number[0]) - 1 ] = 1
X_lables_pure_Matrix = a

# randomize data
indices = np.random.permutation(len(X_data_pure))
X_data_pure_rand = [X_data_pure[i] for i in indices]
Y_lable_rand = [X_lables_pure_Matrix[i] for i in indices]

# separate data
test_length = 250
train_length = len(X_data_pure_rand) - test_length
X_data_train = np.float32(X_data_pure_rand[0: train_length])
X_data_test = np.float32(X_data_pure_rand[train_length : len(X_data_pure_rand)])
X_lables_train = np.float32(Y_lable_rand[0: train_length])
X_lables_test = np.float32(Y_lable_rand[train_length : len(X_data_pure_rand)])

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 25
display_step = 1

# Network Parameters
n_input = 120 # time searies data input (shape: 120 data points)
n_classes = 15 # MNIST total classes (1-15 maturity)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):   
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 120, 1, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=1) # k sollte eigentlich 2 sein aber hier keine 2d Daten

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1) # max(features, 0)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x1 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([15, 1, 1, 32])),
    # 5x1 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([8, 1, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([120*1*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# take patch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        # start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > (train_length / batch_size)):
            batch_number = 0
            r += r
            print('round ', r+1)
        batch_x = next_batch(X_data_train, batch_size, batch_number)
        batch_y = next_batch(X_lables_train, batch_size, batch_number)
        batch_number += batch_number
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Training Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_data_test,
                                      y: X_lables_test,
                                      keep_prob: 1.}))

    
#%%
"""
2000; 5,5; 0.001; Accuracy: 0.3
2000; 5,5; 0.001; Accuracy: 0.24

2000; 10,10; 0.001; Accuracy: 0.288
2000; 10,10; 0.001; Accuracy: 0.296

2000; 12,12; 0.001; Accuracy: 0.276
2000; 12,12; 0.001; Accuracy: 0.26

2000; 15,15; 0.001; Accuracy: 0.408
2000; 15,15; 0.001; Accuracy: 0.332
2000; 15,15; 0.001; Accuracy: 0.368

2000; 20,20; 0.001; Accuracy: 0.312
2000; 20,20; 0.001; Accuracy: 0.312


2000; 15,15; 0.01; Accuracy: 0.984
2000; 15,15; 0.01; Accuracy: 0.184
2000; 15,15; 0.01; Accuracy: 0.084
2000; 15,15; 0.01; Accuracy: 0.564

2500; 15,15; 0.03; Accuracy: 0.432

20000; 15,15; 0.003; Accuracy: 0.232

20000; 15,8; 0.001; Accuracy: 0.408


"""
#%%






def next_batch(x, batch_length, batch_number):
    return x[ batch_number * batch_length : batch_number * batch_length + batch_length]
    
    
a = np.array([[3,3],[4,4],[2,2],[1,1]])
b = np.transpose(a)
print(next_batch(a, 2,1))
batch_x, batch_y = b







#%%
from __future__ import print_function

import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 100
display_step = 5

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:250],
                                      y: mnist.test.labels[:250],
                                      keep_prob: 1.}))
    
#%%
 
    
    