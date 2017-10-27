#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:56:09 2017

@author: Chris
"""

import tensorflow as tf


def Model(n_classes, learning_rate, n_steps, n_input, n_hidden):
    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    # Create model
    def RNN(x, weights, biases):
    
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)
    
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name = "weigths_out")
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]), name = "biases_out")
    }
        
    # Construct model
    pred = RNN(x, weights, biases)
    
    # scaling to avoid the very big output of matrix multiplications in conv_net
    # pred_scaled = pred / 1000
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
       
    return init, optimizer, cost, accuracy, x, y, weights, biases, pred, tf.nn.softmax(pred), correct_pred
