#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:42:39 2017

@author: Chris

A Convolutional Network implementation using TensorFlow library.

working directory is '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/CNN'
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from CNN_Model import Model
cwd = os.getcwd()
from imp import load_source
usefullFunctions = load_source("usefullFunctions", cwd + '/../usefullFunctions.py') 
    
##### Hyperparameter

# Parameters
learning_rate = 0.001
training_iters = 8000
batch_size = 500
display_step = 10 # number of batches until next console output

# Network Parameters
keep_probability = 0.75 # probability to keep units for dropout regularization
conv1Kernal_size = 15
conv1Feature_Number = 8
conv2Kernal_size = 30
conv2Feature_Number = 16

L2_regularization_parameter = 0.7

timeSeriesLength = 360  # time series data input shape: 120 data points

labelNumber, n_classes = usefullFunctions.label_extraction_prams("maturity")

##### Hyperparameter End

path_trainSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand_train.csv'
path_devSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand_dev.csv'
path_testSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand_test.csv'

path_saveModel = cwd + "/trainedModels/CNN-16-10-2017.ckpt"

saveModel = False
plotCostGraph = True
withDevEvaluation = True
withtestAccuracy = True

tf.reset_default_graph()

init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, CNN_output_tf, softmax_tf, correct_pred = Model(n_classes, conv1Feature_Number, conv2Feature_Number, conv1Kernal_size, conv2Kernal_size, learning_rate, timeSeriesLength, L2_regularization_parameter)

### take batch
def next_batch(reader, size, withlables = False, n_classes = 0, normalize = False):
    stream = reader.get_chunk(size)
    # separate labels
    if withlables:
        datapoints  = stream.iloc[:, : stream.shape[1] - 10 ].values
        lables      = stream.iloc[:, stream.shape[1] - labelNumber : stream.shape[1] - (labelNumber-1) ].values
        # create one hot vector for labels
        a = np.zeros((datapoints.shape[0], n_classes)) 
        for i, number in enumerate(lables):
            a[i][ int(number[0]) ] = 1            
        lables = a
    else:
        datapoints  = stream.iloc.values
        lables = None
        
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    return datapoints, lables
    
reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)

if withDevEvaluation:
    reader_dev = pd.read_csv(path_devSet, header=None, sep=";", chunksize=5000, iterator=True)
    x_dev, y_dev = next_batch(reader, 5000, True, n_classes)
    accuracys_dev = []
    
costs = []
iterations = []
accuracys = []

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    layer_dims = [timeSeriesLength, conv1Feature_Number, conv2Feature_Number, 512, int(512/2), int(512/4), int(512/8), int(512/16), n_classes]
    print("Network variables initialized (>", conv1Kernal_size * layer_dims[1] + conv2Kernal_size * conv1Feature_Number * layer_dims[2] + int(timeSeriesLength/4) * layer_dims[2] *  layer_dims[3] + layer_dims[3] * layer_dims[4] + layer_dims[4] * layer_dims[5] + layer_dims[5] * layer_dims[6] + layer_dims[6] * layer_dims[7] + layer_dims[7] * layer_dims[8], ")")
    step = 1
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size <= training_iters:
        
        try:
            batch_x, batch_y = next_batch(reader, batch_size, True, n_classes)
        except StopIteration:
            reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)
            batch_x, batch_y = next_batch(reader, batch_size, True, n_classes)

        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability})
        
        if (step % display_step == 0 or step == 1):
            # Calculate batch loss and accuracy
            loss, acc, softmax = sess.run([cost, accuracy, softmax_tf], feed_dict={x: batch_x,
                                     y: batch_y,
                                     keep_prob: 1.})
    
            if withDevEvaluation:
                acc_dev, correct_pred_dev = sess.run([accuracy, correct_pred], feed_dict={x: x_dev, y: y_dev, keep_prob: 1.})
                accuracys_dev.append(acc_dev)
                 
            # collect data for plots
            costs.append(loss)
            iterations.append(step*batch_size)
            accuracys.append(acc)
        
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            if plotCostGraph:
                if withDevEvaluation:
                    usefullFunctions.plotResults(iterations, costs, accuracys, training_iters, accuracys_dev=accuracys_dev, withDevEvaluation=withDevEvaluation)
                else:
                    usefullFunctions.plotResults(iterations, costs, accuracys, training_iters)
        step += 1
    print("Training Finished!")

    if saveModel:
        saver = tf.train.Saver()
        save_path = saver.save(sess, path_saveModel)
        print("Model saved in file: %s" % save_path)
        
    # stream a test batch
    reader = pd.read_csv(path_testSet, header=None, sep=";", chunksize=5000, iterator=True)
    x_test, y_test = next_batch(reader, 5000, True, n_classes)
    
    # Calculate accuracy for test data
    accuracy_test, correct_pred_testData = sess.run([accuracy, correct_pred], feed_dict={x: x_test, y: y_test, keep_prob: 1.})
    print("Batch Accuracy:", accuracys[-1])
    print("Dev Accuracy:", accuracys_dev[-1])
    print("Testing Accuracy:", accuracy_test)

if plotCostGraph:
    usefullFunctions.plotResults(iterations, costs, accuracys, training_iters, accuracys_dev=accuracys_dev, withDevEvaluation=withDevEvaluation, withtestAccuracy=withtestAccuracy, testAccuracy=accuracy_test)

print("Number of representations in classes (test data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, y_test.shape[0]) if (correct_pred_testData[k] == True and y_test[k,i] == 1))
    number = sum(np.transpose(y_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")

print(" ")

print("Number of representations in classes (dev data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, y_dev.shape[0]) if (correct_pred_dev[k] == True and y_dev[k,i] == 1))
    number = sum(np.transpose(y_dev)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")
