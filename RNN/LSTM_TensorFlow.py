#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:42:39 2017

@author: Chris

A Convolutional Network implementation using TensorFlow library.
"""

# first define results, but only one time. Than all the results are archived there
resultsRNN = [{}]
#%%
#### Functions ####

def randomizeRows(VectData, VectLabel):
    # brings the rows of both vectors in the same new random order
    # both vectors need same length
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return tempData, tempLabel

def randomizeRowsUnlabled(VectData):
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    return tempData
    
def loadDataFromCSV(path, normalize, randomize, lableNumber): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    ###### select here the lables to be analyzed
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    lables = completeData.iloc[:, completeData.shape[1] - lableNumber : completeData.shape[1] - (lableNumber-1) ].values
    ###### select here the lables to be analyzed

    a = np.zeros((datapoints.shape[0], n_classes)) 
    for i, number in enumerate(lables):
        a[i][ int(number[0]) ] = 1        

    lables_Matrix = a
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, lables_rand_list = randomizeRows(datapoints, lables_Matrix)
        datapoints_rand = np.float32(datapoints_rand_list)
        lables_rand = np.float32(lables_rand_list)
    else: 
        datapoints_rand = np.float32(datapoints)
        lables_rand = np.float32(lables_Matrix)
    print('Data loaded')
    return datapoints_rand, lables_rand

def loadDataFromCSVunlabled(path, normalize, randomize): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:][:].values
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, lables_rand_list = randomizeRowsUnlabled(datapoints)
        datapoints_rand = np.float32(datapoints_rand_list)
    else: 
        datapoints_rand = np.float32(datapoints)
        
    print('Data loaded')
    return datapoints_rand
    
def defModel():
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
    
    savedWeights = {
        'out': np.zeros((n_hidden, n_classes))
    }
    savedBiases = {
        'out': np.zeros((n_classes))
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
       
    return init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, pred
    
#%%

import tensorflow as tf
import pandas as pd
import numpy as np
# import math

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 100
display_step = 10

# Network Parameters
n_steps = 360
n_input = 1 # time series data input shape: 360 data points
n_hidden = 128 # hidden layer num of features

timeSeriesLength = 360  # time series data input shape: 360 data points
lableNumber = 7
n_classes = 3
## lables:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

#%%
file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
file2Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Data from CIP/Exposure_0.05Quantile_29052017_11_quarterly.csv'

X, lables = loadDataFromCSV(file1Name, False, True, lableNumber) # normalize?, randomize ? 
X_other, lables_other = loadDataFromCSV(file2Name, False, False, lableNumber)

# separate data in test & training
test_length = int(len(X) * 0.1) # ...% test data
train_length = len(X) - test_length
X_train = np.float32(X[0: train_length])
X_test = np.float32(X[train_length : len(X)])
lables_train = np.float32(lables[0: train_length])
lables_test = np.float32(lables[train_length : len(X)])
print('Data separated')    
    
init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, pred = defModel()
    
# take batch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized")    
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        # start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0
            r += 1
            # new order of rows of data for next trainings run
            randomizeRows(X_train, lables_train)
            print('round ', r+1, "   ", step)
        batch_x = next_batch(X_train, batch_size, batch_number)
        batch_y = next_batch(lables_train, batch_size, batch_number)
        batch_number += batch_number
        batch_x = batch_x.reshape((batch_size, n_steps, n_input)) # z.B. 100x360x1 oder 100x180x2
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
        if(step * batch_size >= training_iters - batch_size):
            with tf.variable_scope('myLSTM'):
                savedWeights['out'] = weights['out'].eval()
                savedBiases['out'] = biases['out'].eval()
            print("Weights saved")            
            
    print("Training Finished!")

    # Calculate accuracy for test data

    TestAtEnd = float(sess.run(accuracy, feed_dict={x: X_test.reshape((-1, n_steps, n_input)),
                                    y: lables_test}))
    print("Testing Accuracy:", TestAtEnd)

    OtherTest = float(sess.run(accuracy, feed_dict={x: X_other.reshape((-1, n_steps, n_input)),
                                                    y: lables_other}))
    print("Other Testing Accuracy:", OtherTest)
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/RNN/savedModels/RNN-test.ckpt")
    print("Model saved in file: %s" % save_path)
    
    
# save the run in dict
newEntry = {"test_accuracy" : TestAtEnd,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "OtherTest_Accuracy" : OtherTest,
            "n_classes" : n_classes,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "comment" : 'basic LSTM',  
            "n_steps" : n_steps,
            "n_hidden" : n_hidden}            
resultsRNN.append(newEntry)


#%%

tf.reset_default_graph()

CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'

X_pred, lables_pred = loadDataFromCSV(CSV_Name, False, False, lableNumber)

init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, pred = defModel()

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/RNN/savedModels/RNN-test.ckpt")
    print("Session restored")
    logits, accur = sess.run([pred, accuracy], feed_dict={   x: X_pred.reshape((-1, n_steps, n_input)), 
                                                             y: lables_pred})
    print("Accuracy:", accur)
    print("Probabilities calculated")  
    logits_scaled = logits / 100000 # das scaling an dieser Stelle verändert die Wkeiten stark, aber nicht die Reihenfolge
    probabilities = tf.nn.softmax(logits_scaled).eval()

# Stack infos together
probabilities_argmax = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((probabilities_argmax, probabilities))

# print the probabilities into a xlsx
import xlsxwriter
workbook = xlsxwriter.Workbook(CSV_Name[:-4] + '_evaluated_RNN.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')


#%%
