#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:33:42 2017

@author: Chris

working directory is '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/CNN'
"""

import tensorflow as tf
import numpy as np

def Model(n_classes, conv1Feature_Number, conv2Feature_Number, conv1Kernal_size, conv2Kernal_size, learning_rate = 0.001, timeSeriesLength = 360, lambd=0.1):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, timeSeriesLength])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    
    # Create some wrappers for simplicity
    def conv1d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and rectivied linear unit (non-linear activation)
        # see padding explanation here http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    
    def maxpool1d(x, k=2):   
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                              padding='SAME')
    
    
    # Create model
    def conv_net(x, weights, biases, keep_prob):
        # Reshape input
        x = tf.reshape(x, shape=[-1, timeSeriesLength, 1])
    
        # Convolution Layer
        conv1 = conv1d(x, weights['wc1'], biases['bc1'], strides = 1 )

        # Max Pooling (down-sampling)
        conv1 = tf.reshape(conv1, shape=[-1, timeSeriesLength, 1, conv1Feature_Number])
        maxp1 = maxpool1d(conv1, k=2)
    #    maxp1 = tf.nn.local_response_normalization(maxp1)
        maxp1 = tf.reshape(maxp1, shape=[-1, int(timeSeriesLength/2), conv1Feature_Number])
    
        # Convolution Layer
        conv2 = conv1d(maxp1, weights['wc2'], biases['bc2'], strides = 1)

        # Max Pooling (down-sampling)
        conv2 = tf.reshape(conv2, shape=[-1, int(timeSeriesLength/2), 1, conv2Feature_Number])
    #    conv2 = tf.nn.local_response_normalization(conv2)
        maxp2 = maxpool1d(conv2, k=2)
        maxp2 = tf.reshape(maxp2, shape=[-1, int(timeSeriesLength/4), conv2Feature_Number]) 
        
        # Fully connected layer 1
        # Reshape maxp2 output to fit fully connected layer input
        fc1 = tf.reshape(maxp2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        relu1 = tf.nn.relu(fc1) # max(x, 0)
        drop1 = tf.nn.dropout(relu1, keep_prob) # Apply Dropout
    
        # Fully connected layer 2
        fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
        relu2 = tf.nn.relu(fc2) # max(x, 0)    
        
        # Fully connected layer 3
        fc3 = tf.add(tf.matmul(relu2, weights['wd3']), biases['bd3'])
        relu3 = tf.nn.relu(fc3) # max(x, 0)    


        # Fully connected layer 4
        fc4 = tf.add(tf.matmul(relu3, weights['wd4']), biases['bd4'])
        relu4 = tf.nn.relu(fc4) # max(x, 0)    

        # Fully connected layer 5
        fc5 = tf.add(tf.matmul(relu4, weights['wd5']), biases['bd5'])
        relu5 = tf.nn.relu(fc5) # max(x, 0)    
    
        # Output, class prediction
        out = tf.add(tf.matmul(relu5, weights['out']), biases['out'])
        return out
    
    ##### Define all the variables    
    fullyConnectedStartSize = 512
    def multiplicationfactor(lastLayerSize):
        return np.sqrt(2/lastLayerSize)
    
    layer_dims = [timeSeriesLength, conv1Feature_Number, conv2Feature_Number, fullyConnectedStartSize, int(fullyConnectedStartSize/2), int(fullyConnectedStartSize/4), int(fullyConnectedStartSize/8), int(fullyConnectedStartSize/16), n_classes]
        
    weights = {
        # 15x1 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv1Kernal_size, 1, layer_dims[1]]) * multiplicationfactor(layer_dims[0])),
        # 10x1 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([conv2Kernal_size, conv1Feature_Number, layer_dims[2]]) * multiplicationfactor(layer_dims[1])),
        # fully connected,30*1*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([int(timeSeriesLength/4)*1*layer_dims[2], layer_dims[3]]) * multiplicationfactor(layer_dims[2])),
        # fully connected, 1024 inputs, 512 outputs        
        'wd2': tf.Variable(tf.random_normal([layer_dims[3], layer_dims[4]]) * multiplicationfactor(layer_dims[3])),
        # fully connected, 512 inputs, 256 outputs        
        'wd3': tf.Variable(tf.random_normal([layer_dims[4], layer_dims[5]]) * multiplicationfactor(layer_dims[4])),
        # fully connected, 256 inputs, 128 outputs        
        'wd4': tf.Variable(tf.random_normal([layer_dims[5], layer_dims[6]]) * multiplicationfactor(layer_dims[5])),
        # fully connected, 256 inputs, 128 outputs        
        'wd5': tf.Variable(tf.random_normal([layer_dims[6], layer_dims[7]]) * multiplicationfactor(layer_dims[6])),
        # 64 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([layer_dims[7], layer_dims[8]]) * multiplicationfactor(layer_dims[7]))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros((conv1Feature_Number))),
        'bc2': tf.Variable(tf.zeros((conv2Feature_Number))),
        'bd1': tf.Variable(tf.zeros((fullyConnectedStartSize))),
        'bd2': tf.Variable(tf.zeros((int(fullyConnectedStartSize/2)))),
        'bd3': tf.Variable(tf.zeros((int(fullyConnectedStartSize/4)))),
        'bd4': tf.Variable(tf.zeros((int(fullyConnectedStartSize/8)))),
        'bd5': tf.Variable(tf.zeros((int(fullyConnectedStartSize/16)))),
        'out': tf.Variable(tf.zeros((n_classes)))
    }
    
    # Construct model
    network_output = conv_net(x, weights, biases, keep_prob) # calculates the probabilities of the different types as array
    
    # L2 regularization (with and without specific TF functions)
    # use : tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights['wc1'])
    # regularizer = tf.contrib.layers.l2_regularizer(lambd)
    # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # L2_regularization_cost_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    L2_regularization_cost_term = lambd/(2*8) * (tf.reduce_sum(tf.square(weights['wc1'])) + tf.reduce_sum(tf.square(weights['wc2'])) + tf.reduce_sum(tf.square(weights['wd1'])) + tf.reduce_sum(tf.square(weights['wd2'])) + tf.reduce_sum(tf.square(weights['wd3'])) + tf.reduce_sum(tf.square(weights['out'])))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y))
    cost_regularization = cost + L2_regularization_cost_term
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_regularization)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(network_output, 1), tf.argmax(y, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, network_output, tf.nn.softmax(network_output), correct_pred
