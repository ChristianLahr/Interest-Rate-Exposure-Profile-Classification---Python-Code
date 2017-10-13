#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:16:08 2017

@author: Chris
"""
import tensorflow as tf
import numpy as np


def maxpool1d(x, dim1, k=2):   

    x = tf.reshape(x, shape=[-1, dim1, 1, 1])
    x = tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')
    x = tf.reshape(x, shape=[-1, int(dim1/k) ])
    return x
        
def Model_easy(networkInput_length, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf = tf.placeholder(tf.float32, [None, networkInput_length])
    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)  
    
    def neuralNetwork(inputData, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData[:,5], shape=[-1, 1])

        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer1) # max(x, 0)
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6), axis = 1)

        fullyConnectedInput = tf.nn.dropout(fullyConnectedInput, keep_prob_tf)    

        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])

#        relu = tf.nn.relu(fc5) # max(x, 0)    
#        maxp1 = maxpool1d(conv1, k=2)    
#        drop = tf.nn.dropout(relu, keep_prob_tf)    
        # Output, class prediction
        out = tf.add(tf.matmul(fullyConnected3, weights['w-out']), biases['b-out'])
        return out
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(6*numberOfNeuronsInFirstLayers), int((6*numberOfNeuronsInFirstLayers)/2) ])),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/2), int((6*numberOfNeuronsInFirstLayers)/4) ])),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/4), int((6*numberOfNeuronsInFirstLayers)/10) ])),
        'w-out': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/10), n_classes]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/2) ])),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/4) ])),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((6*numberOfNeuronsInFirstLayers)/10) ])),
        'b-out': tf.Variable(tf.random_normal([n_classes]))
    }

    prediction = neuralNetwork(inputData_tf, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf, labels_tf, keep_prob_tf, weights, biases, prediction

def Model_easy_RNN(networkInput_length, n_classes, numberOfNeuronsInFirstLayers, learning_rate, n_steps, n_hidden):
   
    rawData_tf = tf.placeholder(tf.float32, [None, 360])
    inputData_tf = tf.placeholder(tf.float32, [None, networkInput_length])
    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)  
    
    def neuralNetwork(rawData, inputData, weights, biases, dropout, networkInput_length, n_steps, n_hidden):

        n_input = 1

        # Reshape input
        inputData_1 = tf.reshape(inputData[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData[:,5], shape=[-1, 1])

        # LSTM Layer
        inputData_LSTM = tf.reshape(rawData, shape=[-1, n_steps, n_input])
        inputData_LSTM = tf.unstack(inputData_LSTM, n_steps, 1) 
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)    # need reuse = True??
        layer_LSTM, states = tf.contrib.rnn.static_rnn(lstm_cell, inputData_LSTM, dtype=tf.float32)     
        layer_LSTM = tf.nn.relu(layer_LSTM[-1]) # max(x, 0)

        # Linear activation, using rnn inner loop last output
        layer_LSTM = tf.add(tf.matmul(layer_LSTM, weights['w-layer-LSTM']), biases['b-layer-LSTM'])
        
        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer1) # max(x, 0)
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6, layer_LSTM), axis = 1)

        fullyConnectedInput = tf.nn.dropout(fullyConnectedInput, keep_prob_tf)    

        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])

        # Output, class prediction
        out = tf.add(tf.matmul(fullyConnected3, weights['w-out']), biases['b-out'])
        return out
        
    fullyConnectedDim = 7*numberOfNeuronsInFirstLayers
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-LSTM': tf.Variable(tf.random_normal([n_hidden, numberOfNeuronsInFirstLayers])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(fullyConnectedDim), int((fullyConnectedDim)/2) ])),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/2), int((fullyConnectedDim)/4) ])),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/4), int((fullyConnectedDim)/10) ])),
        'w-out': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/10), n_classes]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-LSTM': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/2) ])),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/4) ])),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((fullyConnectedDim)/10) ])),
        'b-out': tf.Variable(tf.random_normal([n_classes]))
    }

    prediction = neuralNetwork(rawData_tf, inputData_tf, weights, biases, keep_prob_tf, networkInput_length, n_steps, n_hidden) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Weight normalization
        
    return init, optimizer, cost, accuracy, rawData_tf, inputData_tf, labels_tf, keep_prob_tf, weights, biases, prediction

def Model_together(networkInput_length, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf = tf.placeholder(tf.float32, [None, networkInput_length])
    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)

    def neuralNetwork(inputData, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData, shape=[-1, networkInput_length])

        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
                
        ### Stack Layers together
        fullyConnectedInput = layer1

        fullyConnectedInput = tf.nn.dropout(fullyConnectedInput, keep_prob_tf)    

        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])

#        relu = tf.nn.relu(fc5) # max(x, 0)    
#        maxp1 = maxpool1d(conv1, k=2)    
#        drop = tf.nn.dropout(relu, keep_prob_tf)    
        # Output, class prediction
        out = tf.add(tf.matmul(fullyConnected3, weights['w-out']), biases['b-out'])
        return out
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([networkInput_length, numberOfNeuronsInFirstLayers])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(numberOfNeuronsInFirstLayers), int((numberOfNeuronsInFirstLayers)/2) ])),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/2), int((numberOfNeuronsInFirstLayers)/4) ])),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/4), int((numberOfNeuronsInFirstLayers)/10) ])),
        'w-out': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/10), n_classes]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/2) ])),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/4) ])),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((numberOfNeuronsInFirstLayers)/10) ])),
        'b-out': tf.Variable(tf.random_normal([n_classes]))
    }

    prediction = neuralNetwork(inputData_tf, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf, labels_tf, keep_prob_tf, weights, biases, prediction

def Model_frequency_oneFeature(networkInput_length, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf = tf.placeholder(tf.float32, [None, networkInput_length])
    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)

    def neuralNetwork(inputData, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData[:,3], shape=[-1, 1])

        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        layer1 = maxpool1d(layer1, numberOfNeuronsInFirstLayers, k=2)    
                
        ### Stack Layers together
        fullyConnectedInput = layer1

        fullyConnectedInput = tf.nn.dropout(fullyConnectedInput, keep_prob_tf)    

        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])

#        relu = tf.nn.relu(fc5) # max(x, 0)    
#        maxp1 = maxpool1d(conv1, k=2)    
#        drop = tf.nn.dropout(relu, keep_prob_tf)    
        # Output, class prediction
        out = tf.add(tf.matmul(fullyConnected3, weights['w-out']), biases['b-out'])
        return out
        
    new = int(numberOfNeuronsInFirstLayers/2)
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, int(numberOfNeuronsInFirstLayers)])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(new), int((new)/2) ])),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((new)/2), int((new)/4) ])),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((new)/4), int((new)/10) ])),
        'w-out': tf.Variable(tf.random_normal([ int((new)/10), n_classes]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([int(numberOfNeuronsInFirstLayers)])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int((new)/2) ])),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((new)/4) ])),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((new)/10) ])),
        'b-out': tf.Variable(tf.random_normal([n_classes]))
    }

    prediction = neuralNetwork(inputData_tf, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf, labels_tf, keep_prob_tf, weights, biases, prediction

def Model_Complex(networkInput_length, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf = tf.placeholder(tf.float32, [None, networkInput_length])
    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)

    def neuralNetwork(inputData, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData[:,5], shape=[-1, 1])
        inputData_7 = tf.reshape(inputData[:,6], shape=[-1, 1])
        inputData_8 = tf.reshape(inputData[:,7], shape=[-1, 1])
        inputData_together = tf.reshape(inputData, shape=[-1, networkInput_length])
        
        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        layer1 = maxpool1d(layer1, numberOfNeuronsInFirstLayers, k=2)    
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer2) # max(x, 0)
        layer2 = maxpool1d(layer2, numberOfNeuronsInFirstLayers, k=2)    
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        layer3 = maxpool1d(layer3, numberOfNeuronsInFirstLayers, k=2)    
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        layer4 = maxpool1d(layer4, numberOfNeuronsInFirstLayers, k=2)    
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        layer5 = maxpool1d(layer5, numberOfNeuronsInFirstLayers, k=2)    
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        layer6 = maxpool1d(layer6, numberOfNeuronsInFirstLayers, k=2)    

        layer7 = tf.add(tf.matmul(inputData_7, weights['w-layer-feature7-1']), biases['b-layer-feature7-1'])
        layer7 = tf.nn.relu(layer7) # max(x, 0)
        layer7 = maxpool1d(layer7, numberOfNeuronsInFirstLayers, k=2)    

        layer8 = tf.add(tf.matmul(inputData_8, weights['w-layer-feature8-1']), biases['b-layer-feature8-1'])
        layer8 = tf.nn.relu(layer8) # max(x, 0)
        layer8 = maxpool1d(layer8, numberOfNeuronsInFirstLayers, k=2)    
 
        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=2)    

        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layerTogether), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3, keep_prob_tf)    

        # Output, class prediction
        out = tf.add(tf.matmul(dropout, weights['w-out']), biases['b-out'])
        return out
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature7-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-feature8-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
        'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length, numberOfNeuronsInFirstLayers])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(9*afterMaxPoolDim), int((9*afterMaxPoolDim)/2) ])),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/2), int((9*afterMaxPoolDim)/4) ])),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/4), int((9*afterMaxPoolDim)/10) ])),
        'w-out': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/10), n_classes]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature7-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-feature8-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/2) ])),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/4) ])),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int((9*afterMaxPoolDim)/10) ])),
        'b-out': tf.Variable(tf.random_normal([n_classes]))
    }

    prediction = neuralNetwork(inputData_tf, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf, labels_tf, keep_prob_tf, weights, biases, prediction


def Model_Complex_multiDimFeatures(networkInput_length_oneDimFeatures, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])

    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)

    def neuralNetwork(inputData_oneDimFeatures, inputData_multiDimFeature_jumps, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData_oneDimFeatures[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData_oneDimFeatures[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData_oneDimFeatures[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData_oneDimFeatures[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData_oneDimFeatures[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData_oneDimFeatures[:,5], shape=[-1, 1])
        inputData_7 = tf.reshape(inputData_oneDimFeatures[:,6], shape=[-1, 1])
        inputData_8 = tf.reshape(inputData_oneDimFeatures[:,7], shape=[-1, 1])
        inputData_9 = tf.reshape(inputData_oneDimFeatures[:,8], shape=[-1, 1])
        inputData_10 = tf.reshape(inputData_oneDimFeatures[:,9], shape=[-1, 1])
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        layer1 = maxpool1d(layer1, numberOfNeuronsInFirstLayers, k=2)    
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer2) # max(x, 0)
        layer2 = maxpool1d(layer2, numberOfNeuronsInFirstLayers, k=2)    
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        layer3 = maxpool1d(layer3, numberOfNeuronsInFirstLayers, k=2)    
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        layer4 = maxpool1d(layer4, numberOfNeuronsInFirstLayers, k=2)    
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        layer5 = maxpool1d(layer5, numberOfNeuronsInFirstLayers, k=2)    
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        layer6 = maxpool1d(layer6, numberOfNeuronsInFirstLayers, k=2)    

        layer7 = tf.add(tf.matmul(inputData_7, weights['w-layer-feature7-1']), biases['b-layer-feature7-1'])
        layer7 = tf.nn.relu(layer7) # max(x, 0)
        layer7 = maxpool1d(layer7, numberOfNeuronsInFirstLayers, k=2)    

        layer8 = tf.add(tf.matmul(inputData_8, weights['w-layer-feature8-1']), biases['b-layer-feature8-1'])
        layer8 = tf.nn.relu(layer8) # max(x, 0)
        layer8 = maxpool1d(layer8, numberOfNeuronsInFirstLayers, k=2)    
 
        layer9 = tf.add(tf.matmul(inputData_9, weights['w-layer-feature9-1']), biases['b-layer-feature9-1'])
        layer9 = tf.nn.relu(layer9) # max(x, 0)
        layer9 = maxpool1d(layer9, numberOfNeuronsInFirstLayers, k=2)    

        layer10 = tf.add(tf.matmul(inputData_10, weights['w-layer-feature10-1']), biases['b-layer-feature10-1'])
        layer10 = tf.nn.relu(layer10) # max(x, 0)
        layer10 = maxpool1d(layer10, numberOfNeuronsInFirstLayers, k=2)    

        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=2)    

        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        layer_multiDimFeature_jumps = maxpool1d(layer_multiDimFeature_jumps, int(10*numberOfNeuronsInFirstLayers), k=2)    
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layerTogether, layer_multiDimFeature_jumps), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3, keep_prob_tf)    

        # Output, class prediction
        out = tf.add(tf.matmul(dropout, weights['w-out']), biases['b-out'])
        return out
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
    dimensionFullyConnected = int(11*afterMaxPoolDim) + int((10*numberOfNeuronsInFirstLayers) / 2)
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature7-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature8-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature9-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature10-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers] )),
        'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, int(10*numberOfNeuronsInFirstLayers)] )),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ] )),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ] )),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ] )),
        'w-out': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes] ))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature7-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature8-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature9-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature10-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([int(10*numberOfNeuronsInFirstLayers)] )),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ] )),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ] )),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ] )),
        'b-out': tf.Variable(tf.random_normal([n_classes] ))
    }

    prediction = neuralNetwork(inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, labels_tf, keep_prob_tf, weights, biases, prediction


def Model_Complex_16Features(networkInput_length_oneDimFeatures, n_classes, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_1 = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_multiDimFeature_2 = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_multiDimFeature_3 = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_multiDimFeature_4_358dim = tf.placeholder(tf.float32, [None, 358])

    labels_tf = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)
    
    numberOfNeuronsForMultiDimFeatures = int(6*numberOfNeuronsInFirstLayers)

    def neuralNetwork(inputData_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, weights, biases, dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData_oneDimFeatures[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData_oneDimFeatures[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData_oneDimFeatures[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData_oneDimFeatures[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData_oneDimFeatures[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData_oneDimFeatures[:,5], shape=[-1, 1])
        inputData_7 = tf.reshape(inputData_oneDimFeatures[:,6], shape=[-1, 1])
        inputData_8 = tf.reshape(inputData_oneDimFeatures[:,7], shape=[-1, 1])
        inputData_9 = tf.reshape(inputData_oneDimFeatures[:,8], shape=[-1, 1])
        inputData_10 = tf.reshape(inputData_oneDimFeatures[:,9], shape=[-1, 1])
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        layer1 = maxpool1d(layer1, numberOfNeuronsInFirstLayers, k=2)    
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer2) # max(x, 0)
        layer2 = maxpool1d(layer2, numberOfNeuronsInFirstLayers, k=2)    
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        layer3 = maxpool1d(layer3, numberOfNeuronsInFirstLayers, k=2)    
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        layer4 = maxpool1d(layer4, numberOfNeuronsInFirstLayers, k=2)    
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        layer5 = maxpool1d(layer5, numberOfNeuronsInFirstLayers, k=2)    
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        layer6 = maxpool1d(layer6, numberOfNeuronsInFirstLayers, k=2)    

        layer7 = tf.add(tf.matmul(inputData_7, weights['w-layer-feature7-1']), biases['b-layer-feature7-1'])
        layer7 = tf.nn.relu(layer7) # max(x, 0)
        layer7 = maxpool1d(layer7, numberOfNeuronsInFirstLayers, k=2)    

        layer8 = tf.add(tf.matmul(inputData_8, weights['w-layer-feature8-1']), biases['b-layer-feature8-1'])
        layer8 = tf.nn.relu(layer8) # max(x, 0)
        layer8 = maxpool1d(layer8, numberOfNeuronsInFirstLayers, k=2)    
 
        layer9 = tf.add(tf.matmul(inputData_9, weights['w-layer-feature9-1']), biases['b-layer-feature9-1'])
        layer9 = tf.nn.relu(layer9) # max(x, 0)
        layer9 = maxpool1d(layer9, numberOfNeuronsInFirstLayers, k=2)    

        layer10 = tf.add(tf.matmul(inputData_10, weights['w-layer-feature10-1']), biases['b-layer-feature10-1'])
        layer10 = tf.nn.relu(layer10) # max(x, 0)
        layer10 = maxpool1d(layer10, numberOfNeuronsInFirstLayers, k=2)    

        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=2)    

        layer_multiDimFeature_1 = tf.add(tf.matmul(inputData_tf_multiDimFeature_1, weights['w-layer-multiDimFeature-1']), biases['b-layer-multiDimFeature-1'])
        layer_multiDimFeature_1 = tf.nn.relu(layer_multiDimFeature_1) # max(x, 0)
        layer_multiDimFeature_1 = maxpool1d(layer_multiDimFeature_1, numberOfNeuronsForMultiDimFeatures, k=2)    

        layer_multiDimFeature_2 = tf.add(tf.matmul(inputData_tf_multiDimFeature_2, weights['w-layer-multiDimFeature-2']), biases['b-layer-multiDimFeature-2'])
        layer_multiDimFeature_2 = tf.nn.relu(layer_multiDimFeature_2) # max(x, 0)
        layer_multiDimFeature_2 = maxpool1d(layer_multiDimFeature_2, numberOfNeuronsForMultiDimFeatures, k=2)    

        layer_multiDimFeature_3 = tf.add(tf.matmul(inputData_tf_multiDimFeature_3, weights['w-layer-multiDimFeature-3']), biases['b-layer-multiDimFeature-3'])
        layer_multiDimFeature_3 = tf.nn.relu(layer_multiDimFeature_3) # max(x, 0)
        layer_multiDimFeature_3 = maxpool1d(layer_multiDimFeature_3, numberOfNeuronsForMultiDimFeatures, k=2)    

        layer_multiDimFeature_4 = tf.add(tf.matmul(inputData_tf_multiDimFeature_4_358dim, weights['w-layer-multiDimFeature-4']), biases['b-layer-multiDimFeature-4'])
        layer_multiDimFeature_4 = tf.nn.relu(layer_multiDimFeature_4) # max(x, 0)
        layer_multiDimFeature_4 = maxpool1d(layer_multiDimFeature_4, numberOfNeuronsForMultiDimFeatures, k=2)    

        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layerTogether, layer_multiDimFeature_1, layer_multiDimFeature_2, layer_multiDimFeature_3, layer_multiDimFeature_4), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])
        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3, keep_prob_tf)    

        # Output, class prediction
        out = tf.add(tf.matmul(dropout, weights['w-out']), biases['b-out'])
        return out
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
    dimensionFullyConnected = int(11*afterMaxPoolDim) + int(4 * numberOfNeuronsForMultiDimFeatures / 2)
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature7-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature8-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature9-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-feature10-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers] )),
        'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers] )),
        'w-layer-multiDimFeature-1': tf.Variable(tf.random_normal([359, numberOfNeuronsForMultiDimFeatures] )),
        'w-layer-multiDimFeature-2': tf.Variable(tf.random_normal([359, numberOfNeuronsForMultiDimFeatures] )),
        'w-layer-multiDimFeature-3': tf.Variable(tf.random_normal([359, numberOfNeuronsForMultiDimFeatures] )),
        'w-layer-multiDimFeature-4': tf.Variable(tf.random_normal([358, numberOfNeuronsForMultiDimFeatures] )),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ] )),
        'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ] )),
        'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ] )),
        'w-out': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes] ))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature7-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature8-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature9-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-feature10-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers] )),
        'b-layer-multiDimFeature-1': tf.Variable(tf.random_normal([ numberOfNeuronsForMultiDimFeatures ] )),
        'b-layer-multiDimFeature-2': tf.Variable(tf.random_normal([ numberOfNeuronsForMultiDimFeatures ] )),
        'b-layer-multiDimFeature-3': tf.Variable(tf.random_normal([ numberOfNeuronsForMultiDimFeatures ] )),
        'b-layer-multiDimFeature-4': tf.Variable(tf.random_normal([ numberOfNeuronsForMultiDimFeatures ] )),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ] )),
        'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ] )),
        'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ] )),
        'b-out': tf.Variable(tf.random_normal([n_classes] ))
    }

    prediction = neuralNetwork(inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, labels_tf, keep_prob_tf, weights, biases, prediction, correct_pred


def Model_Complex_multiDimFeatures_5Bit_binary(networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers, learning_rate):
   
    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])

    labels_realValues_tf = tf.placeholder(tf.float32, [None, 1])
    keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)

    def neuralNetwork(inputData_oneDimFeatures, inputData_multiDimFeature_jumps, weights, biases, dropout):

        # Reshape input
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)

        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layerTogether, layer_multiDimFeature_jumps), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2_bit1 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2-bit1']), biases['b-layer-fullyConnected-2-bit1'])
        fullyConnected2_bit2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2-bit2']), biases['b-layer-fullyConnected-2-bit2'])
        fullyConnected2_bit3 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2-bit3']), biases['b-layer-fullyConnected-2-bit3'])
        fullyConnected2_bit4 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2-bit4']), biases['b-layer-fullyConnected-2-bit4'])
        fullyConnected2_bit5 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2-bit5']), biases['b-layer-fullyConnected-2-bit5'])
        fullyConnected3_bit1 = tf.add(tf.matmul(fullyConnected2_bit1, weights['w-layer-fullyConnected-3-bit1']), biases['b-layer-fullyConnected-3-bit1'])
        fullyConnected3_bit2 = tf.add(tf.matmul(fullyConnected2_bit2, weights['w-layer-fullyConnected-3-bit2']), biases['b-layer-fullyConnected-3-bit2'])
        fullyConnected3_bit3 = tf.add(tf.matmul(fullyConnected2_bit3, weights['w-layer-fullyConnected-3-bit3']), biases['b-layer-fullyConnected-3-bit3'])
        fullyConnected3_bit4 = tf.add(tf.matmul(fullyConnected2_bit4, weights['w-layer-fullyConnected-3-bit4']), biases['b-layer-fullyConnected-3-bit4'])
        fullyConnected3_bit5 = tf.add(tf.matmul(fullyConnected2_bit5, weights['w-layer-fullyConnected-3-bit5']), biases['b-layer-fullyConnected-3-bit5'])
        
        # Dropout fct
        dropout_bit1 = tf.nn.dropout(fullyConnected3_bit1, keep_prob_tf)    
        dropout_bit2 = tf.nn.dropout(fullyConnected3_bit2, keep_prob_tf)    
        dropout_bit3 = tf.nn.dropout(fullyConnected3_bit3, keep_prob_tf)    
        dropout_bit4 = tf.nn.dropout(fullyConnected3_bit4, keep_prob_tf)    
        dropout_bit5 = tf.nn.dropout(fullyConnected3_bit5, keep_prob_tf)    

        # Output, class prediction
        out_bit1 = tf.add(tf.matmul(dropout_bit1, weights['w-out_bit1']), biases['b-out_bit1'])
        out_bit2 = tf.add(tf.matmul(dropout_bit2, weights['w-out_bit2']), biases['b-out_bit2'])
        out_bit3 = tf.add(tf.matmul(dropout_bit3, weights['w-out_bit3']), biases['b-out_bit3'])
        out_bit4 = tf.add(tf.matmul(dropout_bit4, weights['w-out_bit4']), biases['b-out_bit4'])
        out_bit5 = tf.add(tf.matmul(dropout_bit5, weights['w-out_bit5']), biases['b-out_bit5'])
        return out_bit1, out_bit2, out_bit3, out_bit4, out_bit5
        
    dimensionFullyConnected = int(16*numberOfNeuronsInFirstLayers)
        
    weights = {
        # 1 input, 100 outputs
        'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, int(11*numberOfNeuronsInFirstLayers)])),
        'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, int(5*numberOfNeuronsInFirstLayers)])),
        'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ])),
        'w-layer-fullyConnected-2-bit1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
        'w-layer-fullyConnected-2-bit2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
        'w-layer-fullyConnected-2-bit3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
        'w-layer-fullyConnected-2-bit4': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
        'w-layer-fullyConnected-2-bit5': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
        'w-layer-fullyConnected-3-bit1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
        'w-layer-fullyConnected-3-bit2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
        'w-layer-fullyConnected-3-bit3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
        'w-layer-fullyConnected-3-bit4': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
        'w-layer-fullyConnected-3-bit5': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
        'w-out_bit1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 2])),
        'w-out_bit2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 2])),
        'w-out_bit3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 2])),
        'w-out_bit4': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 2])),
        'w-out_bit5': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 2]))
    }
    
    biases = {
        # 64 inputs, 10 outputs (class prediction)
        'b-layer-together-1': tf.Variable(tf.random_normal([int(11*numberOfNeuronsInFirstLayers)])),
        'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([int(5*numberOfNeuronsInFirstLayers)])),
        'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ])),
        'b-layer-fullyConnected-2-bit1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
        'b-layer-fullyConnected-2-bit2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
        'b-layer-fullyConnected-2-bit3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
        'b-layer-fullyConnected-2-bit4': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
        'b-layer-fullyConnected-2-bit5': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
        'b-layer-fullyConnected-3-bit1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
        'b-layer-fullyConnected-3-bit2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
        'b-layer-fullyConnected-3-bit3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
        'b-layer-fullyConnected-3-bit4': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
        'b-layer-fullyConnected-3-bit5': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
        'b-out_bit1': tf.Variable(tf.random_normal([2])),
        'b-out_bit2': tf.Variable(tf.random_normal([2])),
        'b-out_bit3': tf.Variable(tf.random_normal([2])),
        'b-out_bit4': tf.Variable(tf.random_normal([2])),
        'b-out_bit5': tf.Variable(tf.random_normal([2]))
    }

    prediction_bit1, prediction_bit2, prediction_bit3, prediction_bit4, prediction_bit5 = neuralNetwork(inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, weights, biases, keep_prob_tf) # calculates the probabilities of the different types as array
    
    # transform bits into numbers (argmax not differentiable or gradient useless)
    bit1 = tf.argmax(prediction_bit1, 1)
    bit2 = tf.argmax(prediction_bit2, 1)
    bit3 = tf.argmax(prediction_bit3, 1)
    bit4 = tf.argmax(prediction_bit4, 1)
    bit5 = tf.argmax(prediction_bit5, 1)
    

    # alternative to argmax (but also no graph connection):
    """
    zero = tf.constant(1, dtype = tf.float32)
    one = tf.constant(1, dtype = tf.float32)
    def f1(): return tf.constant(1)
    def f2(): return tf.constant(0)

    bit1 = tf.map_fn(lambda x: tf.cond(tf.equal(zero, x[0]) & tf.equal(one, x[1]), f1, f2), prediction_bit1,
                            back_prop=True, dtype=tf.float32)
    bit2 = tf.map_fn(lambda x: tf.cond(tf.equal(zero, x[0]) & tf.equal(one, x[1]), f1, f2), prediction_bit2,
                            back_prop=True, dtype=tf.float32)
    bit3 = tf.map_fn(lambda x: tf.cond(tf.equal(zero, x[0]) & tf.equal(one, x[1]), f1, f2), prediction_bit3,
                            back_prop=True, dtype=tf.float32)
    bit4 = tf.map_fn(lambda x: tf.cond(tf.equal(zero, x[0]) & tf.equal(one, x[1]), f1, f2), prediction_bit4,
                            back_prop=True, dtype=tf.float32)
    bit5 = tf.map_fn(lambda x: tf.cond(tf.equal(zero, x[0]) & tf.equal(one, x[1]), f1, f2), prediction_bit5,
                            back_prop=True, dtype=tf.float32)

    # another nice try to avoide argmax:
    bit1 = 1 - tf.slice(prediction_bit1, [0, 0], [1000, 1])
    bit2 = 1 - tf.slice(prediction_bit2, [0, 0], [1000, 1])
    bit3 = 1 - tf.slice(prediction_bit3, [0, 0], [1000, 1])
    bit4 = 1 - tf.slice(prediction_bit4, [0, 0], [1000, 1])
    bit5 = 1 - tf.slice(prediction_bit5, [0, 0], [1000, 1])

    """

    bit1 = tf.reshape(bit1, shape=[1, -1])
    bit2 = tf.reshape(bit2, shape=[1, -1])
    bit3 = tf.reshape(bit2, shape=[1, -1])
    bit4 = tf.reshape(bit2, shape=[1, -1])
    bit5 = tf.reshape(bit2, shape=[1, -1])
    
    tensorInBitRepresentation = tf.concat((bit1, bit2, bit3, bit4, bit5), axis = 0)

    def bitToNumber(tensorInBitRepresentation):
        fuenf = tf.constant(5, dtype=tf.int64)
        tensorWithNumbers = tf.map_fn(lambda x: tf.reduce_sum(
                            tf.reverse(tensor=x, axis=[0])
                            * 2 ** tf.range( fuenf )), tensorInBitRepresentation,
                            back_prop=True, dtype=tf.float32)
        return tensorWithNumbers

    predictionsAsNumber = bitToNumber(tensorInBitRepresentation)
    
    # Define loss and optimizer
    cost = tf.reduce_sum(predictionsAsNumber) # optimizer can not differentiate the "cast" function, so he can't reach the values in backpropagatoin -->  error
#    cost = tf.nn.softmax(predictionsAsNumber) # optimizer can not differentiate the "cast" function, so he can't reach the values in backpropagatoin -->  error
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(predictionsAsNumber, labels_realValues_tf) 
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor from boolean to tf.float32 (0 or 1) type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, labels_realValues_tf, keep_prob_tf, weights, biases, predictionsAsNumber, bit1, tensorInBitRepresentation
    
    
def Model_deepNetwork(networkInput_length_oneDimFeatures, 
                      n_classes_maturity, 
                      n_classes_frequency, 
                      n_classes_coupon, 
                      numberOfNeuronsInFirstLayers, 
                      learning_rate):

    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_probabilities_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    inputData_tf_probabilities_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    inputData_tf_probabilities_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    
    labels_tf_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    labels_tf_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    labels_tf_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    with tf.name_scope('dropout'):
        keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)
        tf.summary.scalar('dropout_keep_probability', keep_prob_tf)

    def neuralNetwork(inputData_oneDimFeatures, 
                      inputData_multiDimFeature_jumps, 
                      inputData_tf_probabilities_Maturity, 
                      inputData_tf_probabilities_Frequency,
                      inputData_tf_probabilities_Coupon,
                      labels_tf_Maturity,
                      labels_tf_Frequency, 
                      labels_tf_Coupon, 
                      weights, 
                      biases, 
                      dropout):

        # Reshape input
        inputData_1 = tf.reshape(inputData_oneDimFeatures[:,0], shape=[-1, 1])
        inputData_2 = tf.reshape(inputData_oneDimFeatures[:,1], shape=[-1, 1])
        inputData_3 = tf.reshape(inputData_oneDimFeatures[:,2], shape=[-1, 1])
        inputData_4 = tf.reshape(inputData_oneDimFeatures[:,3], shape=[-1, 1])
        inputData_5 = tf.reshape(inputData_oneDimFeatures[:,4], shape=[-1, 1])
        inputData_6 = tf.reshape(inputData_oneDimFeatures[:,5], shape=[-1, 1])
        inputData_7 = tf.reshape(inputData_oneDimFeatures[:,6], shape=[-1, 1])
        inputData_8 = tf.reshape(inputData_oneDimFeatures[:,7], shape=[-1, 1])
        inputData_9 = tf.reshape(inputData_oneDimFeatures[:,8], shape=[-1, 1])
        inputData_10 = tf.reshape(inputData_oneDimFeatures[:,9], shape=[-1, 1])
        
        layer1 = tf.add(tf.matmul(inputData_1, weights['w-layer-feature1-1']), biases['b-layer-feature1-1'])
        layer1 = tf.nn.relu(layer1) # max(x, 0)
        layer1 = maxpool1d(layer1, numberOfNeuronsInFirstLayers, k=2)    
        
        layer2 = tf.add(tf.matmul(inputData_2, weights['w-layer-feature2-1']), biases['b-layer-feature2-1'])
        layer2 = tf.nn.relu(layer2) # max(x, 0)
        layer2 = maxpool1d(layer2, numberOfNeuronsInFirstLayers, k=2)    
        
        layer3 = tf.add(tf.matmul(inputData_3, weights['w-layer-feature3-1']), biases['b-layer-feature3-1'])
        layer3 = tf.nn.relu(layer3) # max(x, 0)
        layer3 = maxpool1d(layer3, numberOfNeuronsInFirstLayers, k=2)    
        
        layer4 = tf.add(tf.matmul(inputData_4, weights['w-layer-feature4-1']), biases['b-layer-feature4-1'])
        layer4 = tf.nn.relu(layer4) # max(x, 0)
        layer4 = maxpool1d(layer4, numberOfNeuronsInFirstLayers, k=2)    
        
        layer5 = tf.add(tf.matmul(inputData_5, weights['w-layer-feature5-1']), biases['b-layer-feature5-1'])
        layer5 = tf.nn.relu(layer5) # max(x, 0)
        layer5 = maxpool1d(layer5, numberOfNeuronsInFirstLayers, k=2)    
        
        layer6 = tf.add(tf.matmul(inputData_6, weights['w-layer-feature6-1']), biases['b-layer-feature6-1'])
        layer6 = tf.nn.relu(layer6) # max(x, 0)
        layer6 = maxpool1d(layer6, numberOfNeuronsInFirstLayers, k=2)    

        layer7 = tf.add(tf.matmul(inputData_7, weights['w-layer-feature7-1']), biases['b-layer-feature7-1'])
        layer7 = tf.nn.relu(layer7) # max(x, 0)
        layer7 = maxpool1d(layer7, numberOfNeuronsInFirstLayers, k=2)    

        layer8 = tf.add(tf.matmul(inputData_8, weights['w-layer-feature8-1']), biases['b-layer-feature8-1'])
        layer8 = tf.nn.relu(layer8) # max(x, 0)
        layer8 = maxpool1d(layer8, numberOfNeuronsInFirstLayers, k=2)    
 
        layer9 = tf.add(tf.matmul(inputData_9, weights['w-layer-feature9-1']), biases['b-layer-feature9-1'])
        layer9 = tf.nn.relu(layer9) # max(x, 0)
        layer9 = maxpool1d(layer9, numberOfNeuronsInFirstLayers, k=2)    

        layer10 = tf.add(tf.matmul(inputData_10, weights['w-layer-feature10-1']), biases['b-layer-feature10-1'])
        layer10 = tf.nn.relu(layer10) # max(x, 0)
        layer10 = maxpool1d(layer10, numberOfNeuronsInFirstLayers, k=2)    

        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        layer_multiDimFeature_jumps = maxpool1d(layer_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps'].get_shape().as_list()[1], k=2)    
                      
        layer_trainedModel_Maturity = tf.add(tf.matmul(inputData_tf_probabilities_Maturity, weights['w-layer-trainedModel-Maturiy']), biases['b-layer-trainedModel-Maturiy'])
        layer_trainedModel_Maturity = tf.nn.relu(layer_trainedModel_Maturity) # max(x, 0)
        layer_trainedModel_Maturity = maxpool1d(layer_trainedModel_Maturity, weights['w-layer-trainedModel-Maturiy'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Frequency = tf.add(tf.matmul(inputData_tf_probabilities_Frequency, weights['w-layer-trainedModel-Frequency']), biases['b-layer-trainedModel-Frequency'])
        layer_trainedModel_Frequency = tf.nn.relu(layer_trainedModel_Frequency) # max(x, 0)
        layer_trainedModel_Frequency = maxpool1d(layer_trainedModel_Frequency, weights['w-layer-trainedModel-Frequency'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Coupon = tf.add(tf.matmul(inputData_tf_probabilities_Coupon, weights['w-layer-trainedModel-Coupon']), biases['b-layer-trainedModel-Coupon'])
        layer_trainedModel_Coupon = tf.nn.relu(layer_trainedModel_Coupon) # max(x, 0)
        layer_trainedModel_Coupon = maxpool1d(layer_trainedModel_Coupon, weights['w-layer-trainedModel-Coupon'].get_shape().as_list()[1], k=2)    
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer_multiDimFeature_jumps, layer_trainedModel_Maturity, layer_trainedModel_Frequency, layer_trainedModel_Coupon), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])
        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])

        fullyConnected3_Maturity = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3_Maturity']), biases['b-layer-fullyConnected-3_Maturity'])
        fullyConnected3_Frequency = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3_Frequency']), biases['b-layer-fullyConnected-3_Frequency'])
        fullyConnected3_Coupon = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3_Coupon']), biases['b-layer-fullyConnected-3_Coupon'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3_Maturity, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Frequency, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Coupon, keep_prob_tf)    

        # Output, class prediction
        out_Maturity = tf.add(tf.matmul(dropout, weights['w-out_Maturity']), biases['b-out_Maturity'])
        out_Frequency = tf.add(tf.matmul(dropout, weights['w-out_Frequency']), biases['b-out_Frequency'])
        out_Coupon = tf.add(tf.matmul(dropout, weights['w-out_Coupon']), biases['b-out_Coupon'])
        
        return out_Maturity, out_Frequency, out_Coupon
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
    dimensionFullyConnected = int(10*afterMaxPoolDim) + int((5+3+3+3)*afterMaxPoolDim)

    w_layer_multiDimFeature_jumps_dim = int(5*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Maturiy_dim = int(3*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Frequency_dim = int(3*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Coupon_dim = int(3*numberOfNeuronsInFirstLayers)

    with tf.name_scope('weights'):
        weights = {
            # 1 input, 100 outputs
            'w-layer-feature1-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature2-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature3-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature4-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature5-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature6-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature7-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature8-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature9-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-feature10-1': tf.Variable(tf.random_normal([1, numberOfNeuronsInFirstLayers])),
            'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([n_classes_maturity, w_layer_trainedModel_Maturiy_dim ])),
            'w-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([n_classes_frequency, w_layer_trainedModel_Frequency_dim ])),
            'w-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([n_classes_coupon, w_layer_trainedModel_Coupon_dim ])),
            'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ])),
            'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-out_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_maturity])),
            'w-out_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_frequency])),
            'w-out_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_coupon]))
        }
        tf.summary.histogram('weightsList', weights['w-layer-feature1-1'])
    with tf.name_scope('biases'):
        biases = {
            # 64 inputs, 10 outputs (class prediction)
            'b-layer-feature1-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature2-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature3-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature4-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature5-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature6-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature7-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature8-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature9-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-feature10-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([ w_layer_trainedModel_Maturiy_dim ])),
            'b-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([ w_layer_trainedModel_Frequency_dim ])),
            'b-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([ w_layer_trainedModel_Coupon_dim  ])),
            'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ])),
            'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-out_Maturity': tf.Variable(tf.random_normal([n_classes_maturity])),
            'b-out_Frequency': tf.Variable(tf.random_normal([n_classes_frequency])),
            'b-out_Coupon': tf.Variable(tf.random_normal([n_classes_coupon]))
        }
        tf.summary.histogram('biasesList', biases['b-layer-feature1-1'])
    with tf.name_scope('rawPredictions'):
        prediction_Maturity, prediction_Frequency, prediction_Coupon = neuralNetwork( inputData_tf_oneDimFeatures, 
                                                                               inputData_tf_multiDimFeature_jumps, 
                                                                               inputData_tf_probabilities_Maturity,
                                                                               inputData_tf_probabilities_Frequency,
                                                                               inputData_tf_probabilities_Coupon,
                                                                               labels_tf_Maturity,
                                                                               labels_tf_Frequency, 
                                                                               labels_tf_Coupon, 
                                                                               weights, 
                                                                               biases, 
                                                                               keep_prob_tf) # calculates the probabilities of the different types as array
        tf.summary.histogram('prediction_Maturity', prediction_Maturity)
        tf.summary.histogram('prediction_Frequency', prediction_Frequency)
        tf.summary.histogram('prediction_Coupon', prediction_Coupon)
                                                                                 
    # Define loss and optimizer
    cost_Maturity = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Maturity, labels=labels_tf_Maturity))
    cost_Frequency = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Frequency, labels=labels_tf_Frequency))
    cost_Coupon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Coupon, labels=labels_tf_Coupon))
    cost = cost_Maturity + cost_Frequency + cost_Coupon
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_pred_Maturity'):
            correct_pred_Maturity = tf.equal(tf.argmax(prediction_Maturity, 1), tf.argmax(labels_tf_Maturity, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
        with tf.name_scope('correct_pred_Frequency'):
            correct_pred_Frequency = tf.equal(tf.argmax(prediction_Frequency, 1), tf.argmax(labels_tf_Frequency, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
        with tf.name_scope('correct_pred_Coupon'):
            correct_pred_Coupon = tf.equal(tf.argmax(prediction_Coupon, 1), tf.argmax(labels_tf_Coupon, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    

        with tf.name_scope('accuracy_Maturity'):
            accuracy_Maturity = tf.reduce_mean(tf.cast(correct_pred_Maturity, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Maturity', accuracy_Maturity)
        with tf.name_scope('accuracy_Frequency'):
            accuracy_Frequency = tf.reduce_mean(tf.cast(correct_pred_Frequency, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Frequency', accuracy_Frequency)
        with tf.name_scope('accuracy_Coupon'):
            accuracy_Coupon = tf.reduce_mean(tf.cast(correct_pred_Coupon, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Coupon', accuracy_Coupon)

    correct_pred_all = tf.concat((correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon), axis = 0)
    accuracy_all = tf.reduce_mean(tf.cast(correct_pred_all, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    merged = tf.summary.merge_all()
        
    def normalizeWeightsAndBiases(weights, biases):    
        for index, key in enumerate(weights):        
            maxAll = tf.reduce_max((tf.reduce_max(weights[list(weights.keys())[index]]), tf.reduce_max(biases[list(biases.keys())[index]])))                               
            weights[list(weights.keys())[index]] = tf.divide(weights[list(weights.keys())[index]], maxAll)
            biases[list(biases.keys())[index]] = tf.divide(biases[list(biases.keys())[index]], maxAll)
        return weights, biases
    weights, biases = normalizeWeightsAndBiases(weights, biases)  
    
    return init, optimizer, cost, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf_Maturity, labels_tf_Frequency, labels_tf_Coupon, keep_prob_tf, weights, biases, prediction_Maturity, prediction_Frequency, prediction_Coupon, merged


def Model_deepNetwork_shorter(networkInput_length_oneDimFeatures, 
                              n_classes_maturity, 
                              n_classes_frequency, 
                              n_classes_coupon, 
                              numberOfNeuronsInFirstLayers, 
                              learning_rate):

    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_probabilities_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    inputData_tf_probabilities_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    inputData_tf_probabilities_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    
    labels_tf_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    labels_tf_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    labels_tf_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    with tf.name_scope('dropout'):
        keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)
        tf.summary.scalar('dropout_keep_probability', keep_prob_tf)

    def neuralNetwork(inputData_oneDimFeatures, 
                      inputData_multiDimFeature_jumps, 
                      inputData_tf_probabilities_Maturity, 
                      inputData_tf_probabilities_Frequency,
                      inputData_tf_probabilities_Coupon,
                      labels_tf_Maturity,
                      labels_tf_Frequency, 
                      labels_tf_Coupon, 
                      weights, 
                      biases,
                      k,
                      dropout):
        
        # Reshape input
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        layer_multiDimFeature_jumps = maxpool1d(layer_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps'].get_shape().as_list()[1], k=k)    
                      
        layer_trainedModel_Maturity = tf.add(tf.matmul(inputData_tf_probabilities_Maturity, weights['w-layer-trainedModel-Maturiy']), biases['b-layer-trainedModel-Maturiy'])
        layer_trainedModel_Maturity = tf.nn.relu(layer_trainedModel_Maturity) # max(x, 0)
        layer_trainedModel_Maturity = maxpool1d(layer_trainedModel_Maturity, weights['w-layer-trainedModel-Maturiy'].get_shape().as_list()[1], k=k)    

        layer_trainedModel_Frequency = tf.add(tf.matmul(inputData_tf_probabilities_Frequency, weights['w-layer-trainedModel-Frequency']), biases['b-layer-trainedModel-Frequency'])
        layer_trainedModel_Frequency = tf.nn.relu(layer_trainedModel_Frequency) # max(x, 0)
        layer_trainedModel_Frequency = maxpool1d(layer_trainedModel_Frequency, weights['w-layer-trainedModel-Frequency'].get_shape().as_list()[1], k=k)    

        layer_trainedModel_Coupon = tf.add(tf.matmul(inputData_tf_probabilities_Coupon, weights['w-layer-trainedModel-Coupon']), biases['b-layer-trainedModel-Coupon'])
        layer_trainedModel_Coupon = tf.nn.relu(layer_trainedModel_Coupon) # max(x, 0)
        layer_trainedModel_Coupon = maxpool1d(layer_trainedModel_Coupon, weights['w-layer-trainedModel-Coupon'].get_shape().as_list()[1], k=k)    
        
        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=k)    
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layerTogether, layer_multiDimFeature_jumps, layer_trainedModel_Maturity, layer_trainedModel_Frequency, layer_trainedModel_Coupon), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])

        fullyConnected2_Maturity = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Maturity']), biases['b-layer-fullyConnected-2_Maturity'])
        fullyConnected2_Frequency = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Frequency']), biases['b-layer-fullyConnected-2_Frequency'])
        fullyConnected2_Coupon = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Coupon']), biases['b-layer-fullyConnected-2_Coupon'])

        fullyConnected3_Maturity = tf.add(tf.matmul(fullyConnected2_Maturity, weights['w-layer-fullyConnected-3_Maturity']), biases['b-layer-fullyConnected-3_Maturity'])
        fullyConnected3_Frequency = tf.add(tf.matmul(fullyConnected2_Frequency, weights['w-layer-fullyConnected-3_Frequency']), biases['b-layer-fullyConnected-3_Frequency'])
        fullyConnected3_Coupon = tf.add(tf.matmul(fullyConnected2_Coupon, weights['w-layer-fullyConnected-3_Coupon']), biases['b-layer-fullyConnected-3_Coupon'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3_Maturity, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Frequency, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Coupon, keep_prob_tf)    

        # Output, class prediction
        out_Maturity = tf.add(tf.matmul(dropout, weights['w-out_Maturity']), biases['b-out_Maturity'])
        out_Frequency = tf.add(tf.matmul(dropout, weights['w-out_Frequency']), biases['b-out_Frequency'])
        out_Coupon = tf.add(tf.matmul(dropout, weights['w-out_Coupon']), biases['b-out_Coupon'])
        
        return out_Maturity, out_Frequency, out_Coupon
        
    # parameter for maxpool
    k = 2
    # weigth dimensions
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / k )
    dimensionFullyConnected = int(1*afterMaxPoolDim) + int((3+1+1+1)*afterMaxPoolDim)

    w_layer_multiDimFeature_jumps_dim = int(3*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Maturiy_dim = w_layer_trainedModel_Frequency_dim = w_layer_trainedModel_Coupon_dim = int(numberOfNeuronsInFirstLayers)

    with tf.name_scope('weights'):
        weights = {
            # 1 input, 100 outputs
            'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers])),
            'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([n_classes_maturity, w_layer_trainedModel_Maturiy_dim ])),
            'w-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([n_classes_frequency, w_layer_trainedModel_Frequency_dim ])),
            'w-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([n_classes_coupon, w_layer_trainedModel_Coupon_dim ])),
            'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ])),
            'w-layer-fullyConnected-2_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-2_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-2_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-out_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_maturity])),
            'w-out_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_frequency])),
            'w-out_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_coupon]))
        }
        tf.summary.histogram('layer-together', weights['w-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', weights['w-layer-fullyConnected-1'])
    with tf.name_scope('biases'):
        biases = {
            # 64 inputs, 10 outputs (class prediction)
            'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([ w_layer_trainedModel_Maturiy_dim ])),
            'b-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([ w_layer_trainedModel_Frequency_dim ])),
            'b-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([ w_layer_trainedModel_Coupon_dim  ])),
            'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ])),
            'b-layer-fullyConnected-2_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-2_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-2_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-out_Maturity': tf.Variable(tf.random_normal([n_classes_maturity])),
            'b-out_Frequency': tf.Variable(tf.random_normal([n_classes_frequency])),
            'b-out_Coupon': tf.Variable(tf.random_normal([n_classes_coupon]))
        }
        tf.summary.histogram('layer-together', biases['b-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', biases['b-layer-fullyConnected-1'])
    with tf.name_scope('rawPredictions'):
        prediction_Maturity, prediction_Frequency, prediction_Coupon = neuralNetwork( inputData_tf_oneDimFeatures, 
                                                                               inputData_tf_multiDimFeature_jumps, 
                                                                               inputData_tf_probabilities_Maturity,
                                                                               inputData_tf_probabilities_Frequency,
                                                                               inputData_tf_probabilities_Coupon,
                                                                               labels_tf_Maturity,
                                                                               labels_tf_Frequency, 
                                                                               labels_tf_Coupon, 
                                                                               weights, 
                                                                               biases, 
                                                                               k,
                                                                               keep_prob_tf) # calculates the probabilities of the different types as array
        tf.summary.histogram('prediction_Maturity', prediction_Maturity)
        tf.summary.histogram('prediction_Frequency', prediction_Frequency)
        tf.summary.histogram('prediction_Coupon', prediction_Coupon)
                                                                                 
    # Define loss and optimizer
    cost_Maturity = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Maturity, labels=labels_tf_Maturity))
    cost_Frequency = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Frequency, labels=labels_tf_Frequency))
    cost_Coupon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Coupon, labels=labels_tf_Coupon))
    cost = cost_Maturity + cost_Frequency + cost_Coupon
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_pred_Maturity'):
            predicted_Maturity_Class = tf.argmax(prediction_Maturity, 1)
            correct_pred_Maturity = tf.cast(tf.equal(predicted_Maturity_Class, tf.argmax(labels_tf_Maturity, 1)), tf.float32) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans] der dann zum Typ 0/1 geändert wird    
        with tf.name_scope('correct_pred_Frequency'):
            predicted_Frequency_Class = tf.argmax(prediction_Frequency, 1)
            correct_pred_Frequency = tf.cast(tf.equal(predicted_Frequency_Class, tf.argmax(labels_tf_Frequency, 1)), tf.float32) 
        with tf.name_scope('correct_pred_Coupon'):
            predicted_Coupon_Class = tf.argmax(prediction_Coupon, 1)
            correct_pred_Coupon = tf.cast(tf.equal(predicted_Coupon_Class, tf.argmax(labels_tf_Coupon, 1)), tf.float32) 

        with tf.name_scope('accuracy_Maturity'):
            accuracy_Maturity = tf.reduce_mean(correct_pred_Maturity) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Maturity', accuracy_Maturity)
        with tf.name_scope('accuracy_Frequency'):
            accuracy_Frequency = tf.reduce_mean(correct_pred_Frequency) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Frequency', accuracy_Frequency)
        with tf.name_scope('accuracy_Coupon'):
            accuracy_Coupon = tf.reduce_mean(correct_pred_Coupon) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Coupon', accuracy_Coupon)

    correct_pred_all = tf.concat((correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon), axis = 0)
    accuracy_all = tf.reduce_mean(correct_pred_all) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    merged = tf.summary.merge_all()
        
    def normalizeWeightsAndBiases(weights, biases):    
        for index, key in enumerate(weights):        
            maxAll = tf.reduce_max((tf.reduce_max(weights[list(weights.keys())[index]]), tf.reduce_max(biases[list(biases.keys())[index]])))                               
            weights[list(weights.keys())[index]] = tf.divide(weights[list(weights.keys())[index]], maxAll)
            biases[list(biases.keys())[index]] = tf.divide(biases[list(biases.keys())[index]], maxAll)
        return weights, biases
    weights, biases = normalizeWeightsAndBiases(weights, biases)  
    
    return init, optimizer, cost, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf_Maturity, labels_tf_Frequency, labels_tf_Coupon, keep_prob_tf, weights, biases, predicted_Maturity_Class, predicted_Frequency_Class, predicted_Coupon_Class, correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon, merged

    
def Model_deepNetwork_moreMultiDimInputs(networkInput_length_oneDimFeatures, 
                              n_classes_maturity, 
                              n_classes_frequency, 
                              n_classes_coupon, 
                              numberOfNeuronsInFirstLayers, 
                              learning_rate):

    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_multiDimFeature_delta = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_multiDimFeature_curvature = tf.placeholder(tf.float32, [None, 358])
    inputData_tf_probabilities_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    inputData_tf_probabilities_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    inputData_tf_probabilities_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    
    labels_tf_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    labels_tf_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    labels_tf_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    with tf.name_scope('dropout'):
        keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)
        tf.summary.scalar('dropout_keep_probability', keep_prob_tf)

    def neuralNetwork(inputData_oneDimFeatures, 
                      inputData_multiDimFeature_jumps, 
                      inputData_tf_multiDimFeature_delta,
                      inputData_tf_multiDimFeature_curvature,
                      inputData_tf_probabilities_Maturity, 
                      inputData_tf_probabilities_Frequency,
                      inputData_tf_probabilities_Coupon,
                      labels_tf_Maturity,
                      labels_tf_Frequency, 
                      labels_tf_Coupon, 
                      weights, 
                      biases, 
                      dropout):

        # Reshape input
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        layer_multiDimFeature_jumps = maxpool1d(layer_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps'].get_shape().as_list()[1], k=2)    
                      
        layer_multiDimFeature_delta = tf.add(tf.matmul(inputData_tf_multiDimFeature_delta, weights['w-layer-multiDimFeature-delta']), biases['b-layer-multiDimFeature-delta'])
        layer_multiDimFeature_delta = tf.nn.relu(layer_multiDimFeature_delta) # max(x, 0)
        layer_multiDimFeature_delta = maxpool1d(layer_multiDimFeature_delta, weights['w-layer-multiDimFeature-delta'].get_shape().as_list()[1], k=2)    

        layer_multiDimFeature_curvature = tf.add(tf.matmul(inputData_tf_multiDimFeature_curvature, weights['w-layer-multiDimFeature-curvature']), biases['b-layer-multiDimFeature-curvature'])
        layer_multiDimFeature_curvature = tf.nn.relu(layer_multiDimFeature_curvature) # max(x, 0)
        layer_multiDimFeature_curvature = maxpool1d(layer_multiDimFeature_curvature, weights['w-layer-multiDimFeature-curvature'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Maturity = tf.add(tf.matmul(inputData_tf_probabilities_Maturity, weights['w-layer-trainedModel-Maturiy']), biases['b-layer-trainedModel-Maturiy'])
        layer_trainedModel_Maturity = tf.nn.relu(layer_trainedModel_Maturity) # max(x, 0)
        layer_trainedModel_Maturity = maxpool1d(layer_trainedModel_Maturity, weights['w-layer-trainedModel-Maturiy'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Frequency = tf.add(tf.matmul(inputData_tf_probabilities_Frequency, weights['w-layer-trainedModel-Frequency']), biases['b-layer-trainedModel-Frequency'])
        layer_trainedModel_Frequency = tf.nn.relu(layer_trainedModel_Frequency) # max(x, 0)
        layer_trainedModel_Frequency = maxpool1d(layer_trainedModel_Frequency, weights['w-layer-trainedModel-Frequency'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Coupon = tf.add(tf.matmul(inputData_tf_probabilities_Coupon, weights['w-layer-trainedModel-Coupon']), biases['b-layer-trainedModel-Coupon'])
        layer_trainedModel_Coupon = tf.nn.relu(layer_trainedModel_Coupon) # max(x, 0)
        layer_trainedModel_Coupon = maxpool1d(layer_trainedModel_Coupon, weights['w-layer-trainedModel-Coupon'].get_shape().as_list()[1], k=2)    
        
        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=2)    
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layerTogether, layer_multiDimFeature_jumps, layer_multiDimFeature_delta, layer_multiDimFeature_curvature, layer_trainedModel_Maturity, layer_trainedModel_Frequency, layer_trainedModel_Coupon), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])

        fullyConnected2_Maturity = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Maturity']), biases['b-layer-fullyConnected-2_Maturity'])
        fullyConnected2_Frequency = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Frequency']), biases['b-layer-fullyConnected-2_Frequency'])
        fullyConnected2_Coupon = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2_Coupon']), biases['b-layer-fullyConnected-2_Coupon'])

        fullyConnected3_Maturity = tf.add(tf.matmul(fullyConnected2_Maturity, weights['w-layer-fullyConnected-3_Maturity']), biases['b-layer-fullyConnected-3_Maturity'])
        fullyConnected3_Frequency = tf.add(tf.matmul(fullyConnected2_Frequency, weights['w-layer-fullyConnected-3_Frequency']), biases['b-layer-fullyConnected-3_Frequency'])
        fullyConnected3_Coupon = tf.add(tf.matmul(fullyConnected2_Coupon, weights['w-layer-fullyConnected-3_Coupon']), biases['b-layer-fullyConnected-3_Coupon'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3_Maturity, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Frequency, keep_prob_tf)    
        dropout = tf.nn.dropout(fullyConnected3_Coupon, keep_prob_tf)    

        # Output, class prediction
        out_Maturity = tf.add(tf.matmul(dropout, weights['w-out_Maturity']), biases['b-out_Maturity'])
        out_Frequency = tf.add(tf.matmul(dropout, weights['w-out_Frequency']), biases['b-out_Frequency'])
        out_Coupon = tf.add(tf.matmul(dropout, weights['w-out_Coupon']), biases['b-out_Coupon'])
        
        return out_Maturity, out_Frequency, out_Coupon
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
    dimensionFullyConnected = int(1*afterMaxPoolDim) + int((3+3+3+1+1+1)*afterMaxPoolDim)

    w_layer_multiDimFeature_jumps_dim = int(3*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Maturiy_dim = w_layer_trainedModel_Frequency_dim = w_layer_trainedModel_Coupon_dim = int(numberOfNeuronsInFirstLayers)

    with tf.name_scope('weights'):
        weights = {
            # 1 input, 100 outputs
            'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers])),
            'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-multiDimFeature-delta': tf.Variable(tf.random_normal([359, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-multiDimFeature-curvature': tf.Variable(tf.random_normal([358, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([n_classes_maturity, w_layer_trainedModel_Maturiy_dim ])),
            'w-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([n_classes_frequency, w_layer_trainedModel_Frequency_dim ])),
            'w-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([n_classes_coupon, w_layer_trainedModel_Coupon_dim ])),
            'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ])),
            'w-layer-fullyConnected-2_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-2_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-2_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-out_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_maturity])),
            'w-out_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_frequency])),
            'w-out_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), n_classes_coupon]))
        }
        tf.summary.histogram('layer-together', weights['w-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', weights['w-layer-fullyConnected-1'])
    with tf.name_scope('biases'):
        biases = {
            # 64 inputs, 10 outputs (class prediction)
            'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-multiDimFeature-delta': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-multiDimFeature-curvature': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([ w_layer_trainedModel_Maturiy_dim ])),
            'b-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([ w_layer_trainedModel_Frequency_dim ])),
            'b-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([ w_layer_trainedModel_Coupon_dim  ])),
            'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ])),
            'b-layer-fullyConnected-2_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-2_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-2_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-3_Maturity': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Frequency': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-layer-fullyConnected-3_Coupon': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-out_Maturity': tf.Variable(tf.random_normal([n_classes_maturity])),
            'b-out_Frequency': tf.Variable(tf.random_normal([n_classes_frequency])),
            'b-out_Coupon': tf.Variable(tf.random_normal([n_classes_coupon]))
        }
        tf.summary.histogram('layer-together', biases['b-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', biases['b-layer-fullyConnected-1'])
    with tf.name_scope('rawPredictions'):
        prediction_Maturity, prediction_Frequency, prediction_Coupon = neuralNetwork( inputData_tf_oneDimFeatures, 
                                                                               inputData_tf_multiDimFeature_jumps, 
                                                                               inputData_tf_multiDimFeature_delta,
                                                                               inputData_tf_multiDimFeature_curvature,
                                                                               inputData_tf_probabilities_Maturity,
                                                                               inputData_tf_probabilities_Frequency,
                                                                               inputData_tf_probabilities_Coupon,
                                                                               labels_tf_Maturity,
                                                                               labels_tf_Frequency, 
                                                                               labels_tf_Coupon, 
                                                                               weights, 
                                                                               biases, 
                                                                               keep_prob_tf) # calculates the probabilities of the different types as array
        tf.summary.histogram('prediction_Maturity', prediction_Maturity)
        tf.summary.histogram('prediction_Frequency', prediction_Frequency)
        tf.summary.histogram('prediction_Coupon', prediction_Coupon)
                                                                                 
    # Define loss and optimizer
    cost_Maturity = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Maturity, labels=labels_tf_Maturity))
    cost_Frequency = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Frequency, labels=labels_tf_Frequency))
    cost_Coupon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_Coupon, labels=labels_tf_Coupon))
    cost = cost_Maturity + cost_Frequency + cost_Coupon
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_pred_Maturity'):
            predicted_Maturity_Class = tf.argmax(prediction_Maturity, 1)
            correct_pred_Maturity = tf.cast(tf.equal(predicted_Maturity_Class, tf.argmax(labels_tf_Maturity, 1)), tf.float32) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans] der dann zum Typ 0/1 geändert wird    
        with tf.name_scope('correct_pred_Frequency'):
            predicted_Frequency_Class = tf.argmax(prediction_Frequency, 1)
            correct_pred_Frequency = tf.cast(tf.equal(predicted_Frequency_Class, tf.argmax(labels_tf_Frequency, 1)), tf.float32) 
        with tf.name_scope('correct_pred_Coupon'):
            predicted_Coupon_Class = tf.argmax(prediction_Coupon, 1)
            correct_pred_Coupon = tf.cast(tf.equal(predicted_Coupon_Class, tf.argmax(labels_tf_Coupon, 1)), tf.float32) 

        with tf.name_scope('accuracy_Maturity'):
            
            accuracy_Maturity = tf.reduce_mean(correct_pred_Maturity) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Maturity', accuracy_Maturity)
        with tf.name_scope('accuracy_Frequency'):
            accuracy_Frequency = tf.reduce_mean(correct_pred_Frequency) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Frequency', accuracy_Frequency)
        with tf.name_scope('accuracy_Coupon'):
            accuracy_Coupon = tf.reduce_mean(correct_pred_Coupon) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Coupon', accuracy_Coupon)

    correct_pred_all = tf.concat((correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon), axis = 0)
    accuracy_all = tf.reduce_mean(correct_pred_all) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    merged = tf.summary.merge_all()
        
    def normalizeWeightsAndBiases(weights, biases):    
        for index, key in enumerate(weights):        
            maxAll = tf.reduce_max((tf.reduce_max(weights[list(weights.keys())[index]]), tf.reduce_max(biases[list(biases.keys())[index]])))                               
            weights[list(weights.keys())[index]] = tf.divide(weights[list(weights.keys())[index]], maxAll)
            biases[list(biases.keys())[index]] = tf.divide(biases[list(biases.keys())[index]], maxAll)
        return weights, biases
    weights, biases = normalizeWeightsAndBiases(weights, biases)  
    
    return init, optimizer, cost, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_multiDimFeature_delta, inputData_tf_multiDimFeature_curvature, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf_Maturity, labels_tf_Frequency, labels_tf_Coupon, keep_prob_tf, weights, biases, predicted_Maturity_Class, predicted_Frequency_Class, predicted_Coupon_Class, correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon, merged


def Model_deepNetwork_shorter_72classes(networkInput_length_oneDimFeatures, 
                              n_classes_maturity, 
                              n_classes_frequency, 
                              n_classes_coupon, 
                              numberOfNeuronsInFirstLayers, 
                              learning_rate):

    inputData_tf_oneDimFeatures = tf.placeholder(tf.float32, [None, networkInput_length_oneDimFeatures])
    inputData_tf_multiDimFeature_jumps = tf.placeholder(tf.float32, [None, 359])
    inputData_tf_probabilities_Maturity = tf.placeholder(tf.float32, [None, n_classes_maturity])
    inputData_tf_probabilities_Frequency = tf.placeholder(tf.float32, [None, n_classes_frequency])
    inputData_tf_probabilities_Coupon = tf.placeholder(tf.float32, [None, n_classes_coupon])
    
    labels_tf = tf.placeholder(tf.float32, [None, 72 ])
    
    with tf.name_scope('dropout'):
        keep_prob_tf = tf.placeholder(tf.float32) # dropout (keep probability)
        tf.summary.scalar('dropout_keep_probability', keep_prob_tf)

    def neuralNetwork(inputData_oneDimFeatures, 
                      inputData_multiDimFeature_jumps, 
                      inputData_tf_probabilities_Maturity, 
                      inputData_tf_probabilities_Frequency,
                      inputData_tf_probabilities_Coupon,
                      weights, 
                      biases, 
                      dropout):

        # Reshape input
        inputData_together = tf.reshape(inputData_oneDimFeatures, shape=[-1, networkInput_length_oneDimFeatures])
        
        layer_multiDimFeature_jumps = tf.add(tf.matmul(inputData_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps']), biases['b-layer-multiDimFeature-jumps'])
        layer_multiDimFeature_jumps = tf.nn.relu(layer_multiDimFeature_jumps) # max(x, 0)
        layer_multiDimFeature_jumps = maxpool1d(layer_multiDimFeature_jumps, weights['w-layer-multiDimFeature-jumps'].get_shape().as_list()[1], k=2)    
                      
        layer_trainedModel_Maturity = tf.add(tf.matmul(inputData_tf_probabilities_Maturity, weights['w-layer-trainedModel-Maturiy']), biases['b-layer-trainedModel-Maturiy'])
        layer_trainedModel_Maturity = tf.nn.relu(layer_trainedModel_Maturity) # max(x, 0)
        layer_trainedModel_Maturity = maxpool1d(layer_trainedModel_Maturity, weights['w-layer-trainedModel-Maturiy'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Frequency = tf.add(tf.matmul(inputData_tf_probabilities_Frequency, weights['w-layer-trainedModel-Frequency']), biases['b-layer-trainedModel-Frequency'])
        layer_trainedModel_Frequency = tf.nn.relu(layer_trainedModel_Frequency) # max(x, 0)
        layer_trainedModel_Frequency = maxpool1d(layer_trainedModel_Frequency, weights['w-layer-trainedModel-Frequency'].get_shape().as_list()[1], k=2)    

        layer_trainedModel_Coupon = tf.add(tf.matmul(inputData_tf_probabilities_Coupon, weights['w-layer-trainedModel-Coupon']), biases['b-layer-trainedModel-Coupon'])
        layer_trainedModel_Coupon = tf.nn.relu(layer_trainedModel_Coupon) # max(x, 0)
        layer_trainedModel_Coupon = maxpool1d(layer_trainedModel_Coupon, weights['w-layer-trainedModel-Coupon'].get_shape().as_list()[1], k=2)    
        
        layerTogether = tf.add(tf.matmul(inputData_together, weights['w-layer-together-1']), biases['b-layer-together-1'])
        layerTogether = tf.nn.relu(layerTogether) # max(x, 0)
        layerTogether = maxpool1d(layerTogether, numberOfNeuronsInFirstLayers, k=2)    
        
        ### Stack Layers together
        fullyConnectedInput = tf.concat((layerTogether, layer_multiDimFeature_jumps, layer_trainedModel_Maturity, layer_trainedModel_Frequency, layer_trainedModel_Coupon), axis = 1)

        # Fully connected layers
        fullyConnected1 = tf.add(tf.matmul(fullyConnectedInput, weights['w-layer-fullyConnected-1']), biases['b-layer-fullyConnected-1'])

        fullyConnected2 = tf.add(tf.matmul(fullyConnected1, weights['w-layer-fullyConnected-2']), biases['b-layer-fullyConnected-2'])

        fullyConnected3 = tf.add(tf.matmul(fullyConnected2, weights['w-layer-fullyConnected-3']), biases['b-layer-fullyConnected-3'])
        
        # Dropout fct
        dropout = tf.nn.dropout(fullyConnected3, keep_prob_tf)    

        # Output, class prediction
        out = tf.add(tf.matmul(dropout, weights['w-out']), biases['b-out'])
        
        return out
        
    afterMaxPoolDim = int(numberOfNeuronsInFirstLayers / 2 )
    dimensionFullyConnected = int(1*afterMaxPoolDim) + int((3+1+1+1)*afterMaxPoolDim)

    w_layer_multiDimFeature_jumps_dim = int(3*numberOfNeuronsInFirstLayers)
    w_layer_trainedModel_Maturiy_dim = w_layer_trainedModel_Frequency_dim = w_layer_trainedModel_Coupon_dim = int(numberOfNeuronsInFirstLayers)

    with tf.name_scope('weights'):
        weights = {
            # 1 input, 100 outputs
            'w-layer-together-1': tf.Variable(tf.random_normal([networkInput_length_oneDimFeatures, numberOfNeuronsInFirstLayers])),
            'w-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([359, w_layer_multiDimFeature_jumps_dim ])),
            'w-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([n_classes_maturity, w_layer_trainedModel_Maturiy_dim ])),
            'w-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([n_classes_frequency, w_layer_trainedModel_Frequency_dim ])),
            'w-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([n_classes_coupon, w_layer_trainedModel_Coupon_dim ])),
            'w-layer-fullyConnected-1': tf.Variable(tf.random_normal([ dimensionFullyConnected, int(dimensionFullyConnected/2) ])),
            'w-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2), int(dimensionFullyConnected/4) ])),
            'w-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4), int(dimensionFullyConnected/10) ])),
            'w-out': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10), 72 ])),
        }
        tf.summary.histogram('layer-together', weights['w-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', weights['w-layer-fullyConnected-1'])
    with tf.name_scope('biases'):
        biases = {
            # 64 inputs, 10 outputs (class prediction)
            'b-layer-together-1': tf.Variable(tf.random_normal([numberOfNeuronsInFirstLayers])),
            'b-layer-multiDimFeature-jumps': tf.Variable(tf.random_normal([ w_layer_multiDimFeature_jumps_dim ])),
            'b-layer-trainedModel-Maturiy': tf.Variable(tf.random_normal([ w_layer_trainedModel_Maturiy_dim ])),
            'b-layer-trainedModel-Frequency': tf.Variable(tf.random_normal([ w_layer_trainedModel_Frequency_dim ])),
            'b-layer-trainedModel-Coupon': tf.Variable(tf.random_normal([ w_layer_trainedModel_Coupon_dim  ])),
            'b-layer-fullyConnected-1': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/2) ])),
            'b-layer-fullyConnected-2': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/4) ])),
            'b-layer-fullyConnected-3': tf.Variable(tf.random_normal([ int(dimensionFullyConnected/10) ])),
            'b-out': tf.Variable(tf.random_normal([ 72 ])),
        }
        tf.summary.histogram('layer-together', biases['b-layer-together-1'])
        tf.summary.histogram('fullyConnected-1', biases['b-layer-fullyConnected-1'])
    with tf.name_scope('rawPredictions'):
        prediction = neuralNetwork(   inputData_tf_oneDimFeatures, 
                                               inputData_tf_multiDimFeature_jumps, 
                                               inputData_tf_probabilities_Maturity,
                                               inputData_tf_probabilities_Frequency,
                                               inputData_tf_probabilities_Coupon,
                                               weights, 
                                               biases, 
                                               keep_prob_tf) # calculates the probabilities of the different types as array
        
        tf.summary.histogram('prediction', prediction)
                                                                                 
    # Define loss and optimizer
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_tf))
        tf.summary.scalar('cost', cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    with tf.name_scope('accuracyProcess'):
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
            tf.summary.scalar('accuracy_Maturity', accuracy)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    merged = tf.summary.merge_all()
        
    def normalizeWeightsAndBiases(weights, biases):    
        for index, key in enumerate(weights):        
            maxAll = tf.reduce_max((tf.reduce_max(weights[list(weights.keys())[index]]), tf.reduce_max(biases[list(biases.keys())[index]])))                               
            weights[list(weights.keys())[index]] = tf.divide(weights[list(weights.keys())[index]], maxAll)
            biases[list(biases.keys())[index]] = tf.divide(biases[list(biases.keys())[index]], maxAll)
        return weights, biases
    weights, biases = normalizeWeightsAndBiases(weights, biases)  
    
    return init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf, keep_prob_tf, weights, biases, prediction, merged
            
    
def featureCalculation(timeSeriesLength, numberOfProfiles):
    data = tf.placeholder(tf.float32, [numberOfProfiles, timeSeriesLength])

    print("calclate features 1 & 2")
    negmaximum_tf = tf.reduce_max(-data, axis = 1)
    maximum_tf = tf.reduce_max(data, axis = 1)
    maximum_tf = tf.where(tf.greater(negmaximum_tf, maximum_tf), negmaximum_tf, maximum_tf)
    maximumPosition_tf = tf.argmax(data, axis = 1)

    print("calclate features 3")
    entryPoint_np = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        entryPoint_np[i] = data[i, 0]

    print("calclate features 4")
    deltaMaxFist = tf.subtract( maximum_tf, entryPoint_np )

    print("calclate features 5")
    delta_tf = tf.map_fn(lambda x: tf.map_fn(lambda y: y - (y-1), x, back_prop=True, dtype=tf.float32), 
                            data, back_prop=True, dtype=tf.float32)





    maximum_np = np.zeros(numberOfProfiles, dtype = np.float32)
    maximum = tf.Variable(maximum_np, dtype = np.float32)
    negmaximum_tf = tf.reduce_max(-data, axis = 1)
    maximum_tf = tf.reduce_max(data, axis = 1)
    # assigns the maximum tensor where the condition "greater" is true with negmax and false with max (all are vectors)
    maximum_np = maximum.assign( tf.where(tf.greater(negmaximum_tf, maximum_tf), negmaximum_tf, maximum_tf))
#    tf.assign(maximu, ...)
    
    maximumPosition_np = np.zeros(numberOfProfiles, dtype = np.float32)
    maximumPosition_np = tf.map_fn(lambda x: tf.argmax(x), maximum_np,
                            back_prop=True, dtype=tf.float32)

#    res = tf.Variable(maximumPosition_np, dtype = np.float32)
    res = tf.to_int32(tf.argmax(tf.where(tf.equal(data[0], tf.fill([timeSeriesLength], maximum_np[0])))))
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        print(i)
        res = tf.concat([res, tf.to_int32(tf.argmax(tf.where(tf.equal(data[i], tf.fill([timeSeriesLength], maximum_np[i])))))], axis = 0)
#        indices = [[i]]
 #       values = [tf.argmax(tf.where(tf.equal(data[i], tf.fill([timeSeriesLength], maximum_np[i]))))]
  #      shape = [numberOfProfiles]
   #     delta = tf.SparseTensor(indices, values, shape)
    #    maximumPosition_np = maximumPosition_np + tf.sparse_tensor_to_dense(delta)

#        maximumPosition_np[i] = tf.argmax(tf.where(tf.equal(data[i], constantTensor_np)))
    
        
#        print("qwer", data[i].get_shape())


        
    ## Feature 3: First Point
    print("calclate features 3")
    entryPoint = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        entryPoint[i] = data[i, 0]
    
    ## Feature 4: Delta(maximum, first point)
    print("calclate features 4")
    deltaMaxFist = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        deltaMaxFist[i] = maximum[i] - entryPoint[i]
    
    ## Feature 5: delta to the next point (first derivative)
    print("calclate features 5")
    delta = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        for j in range(0, timeSeriesLength - 1):
            delta[i,j] = data[i,j] - data[i,j+ 1]

    ## Feature 6: max(delta, 0) = down jumps
    print("calclate features 6")
    jumps = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        for j in range(0, timeSeriesLength - 1):
            jumps[i,j] = max(delta[i,j], 0)
    
    ## Feature 7: time passed after last jump
    print("calclate features 7")
    timeAfterJump = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        time = 0
        for j in range(0, timeSeriesLength - 1):
            if(jumps[i,j] == 0): time = time +1
            if(jumps[i,j] > 0): time = 0
            timeAfterJump[i,j] = time
    
    ## Feature 8: number of zeros at the end (length of the zero line at the end)
    print("calclate features 8")
    numberOfZeros = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfZeros[i] = timeAfterJump[i, 358] #there are only 359 jumps --> index 358
        if(data[i,359] != 0): numberOfZeros[i] = 0
    
    ## Feature 9: maximal distance to the next drop3
    print("calclate features 9")
    maximalDistanceToTheNextDrop = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses   
        maximalDistanceToTheNextDrop[i] = max(timeAfterJump[i,int(maximumPosition[i]):int(360-numberOfZeros[i])])
    
        
    ## Feature 10: mean of the tensor without last zeros
    print("calclate features 10")
    mean = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        mean[i] = np.mean(data[i])
    
    ## Feature 11: variance of the tensor without last zeros
    print("calclate features 11")
    variance = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        variance[i] = np.var(data[i])  

    ## Feature 12 & 13: number of jumps and average jump higth
    print("calclate features 12 & 13")
    numberOfJumps = np.zeros(numberOfProfiles, dtype = np.float32)
    averageJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        number_temp = 0
        sum_temp = 0
        for j in range(0, timeSeriesLength - 1):
            if(jumps[i,j] > 0): 
                number_temp += 1
                sum_temp += jumps[i,j]
        numberOfJumps[i] = number_temp
        if(number_temp == 0):   averageJump[i] = 0
        if(number_temp > 0):    averageJump[i] = sum_temp / number_temp

    ## Feature 14: number of jumps relative to net (real) time series length
    print("calclate features 14")
    numberOfJumpsRelativeToLength = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfJumpsRelativeToLength[i] = numberOfJumps[i] / (360-numberOfZeros[i])
    
    ## Feature 15: highest jump
    print("calclate features 15")
    maximalJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        maximalJump[i] = max(jumps[i])
        
    return maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump