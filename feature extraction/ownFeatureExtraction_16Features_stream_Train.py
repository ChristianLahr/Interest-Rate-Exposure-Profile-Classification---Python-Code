#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:51:19 2017

@author: Chris
working directory is '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/feature extraction'
"""

import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_Complex_16Features
import pandas as pd
import os
cwd = os.getcwd()
from imp import load_source
usefullFunctions = load_source("usefullFunctions", cwd + '/../usefullFunctions.py') 

path_trainSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_train.csv'
path_devSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_dev.csv'
path_testSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_test.csv'

### Parameters
learning_rate = 0.001
training_iters = 100000
# batch size should not be bigger than available datasets (because than every batch = all data --> meaningless)
batch_size = 100
display_step = 50
#

numberWeigthsPerNeuronInFirstLayer = 20
keep_probability = 0.75
saveModel = True
path_saveModel = cwd + '/trainedModels/ownFeatureExtraction_16Features_maturity.ckpt'
plotCostGraph = True
withDevEvaluation = True
withtestAccuracy = True


labelNumber, n_classes = usefullFunctions.label_extraction_prams("maturity")

### Be shure that the old graph is desleted
tf.reset_default_graph()
                                       
def next_batch_ofFeatures(reader, size):

    stream = reader.get_chunk(size)
    # separate labels
    features = stream.iloc[:, : stream.shape[1] - 10].values
    labels   = stream.iloc[:, stream.shape[1] - labelNumber : stream.shape[1] - (labelNumber-1) ].values
                                      
    # create one hot vector for labels
    def oneHot(length, n_classes, lab):
        a = np.zeros((features.shape[0], n_classes)) 
        for i, number in enumerate(lab):
            a[i][ int(number[0]) ] = 1        
        return a
            
    labels = oneHot(size, n_classes, labels)
    
    oneDimFeatures                  = features[:,:12]
    # order: [maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump]
    delta                           = features[:,12:371]
    jumps                           = features[:,371:730]
    timeAfterJump                   = features[:,730:1089]
    curvature                       = features[:,1089:1447]
 
    return oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels

n_oneDImFeatures = 12

### define the Model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, labels_tf, keep_prob_tf, weights, biases, prediction, _, correct_pred = Model_Complex_16Features(n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate=learning_rate, networkInput_length_oneDimFeatures=n_oneDImFeatures)

# start the stream
reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)

if withDevEvaluation:
    reader_dev = pd.read_csv(path_devSet, header=None, sep=";", chunksize=5000, iterator=True)
    oneDimFeatures_dev, delta_dev, jumps_dev, timeAfterJump_dev, curvature_dev, labels_dev = next_batch_ofFeatures(reader, 5000)
    accuracys_dev = []

costs = []
iterations = []
accuracys = []

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized")    
    step = 1
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size <= training_iters:
        
        try:
            oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels = next_batch_ofFeatures(reader, batch_size)
        except StopIteration: # restart iteration if data has already passed
            print("restart reader iteration")
            reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)
            oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels = next_batch_ofFeatures(reader, batch_size)
 
        multiDimFeature_1_359dim = jumps
        multiDimFeature_2_359dim = delta
        multiDimFeature_3_359dim = timeAfterJump
        multiDimFeature_4_358dim = curvature

        ### Run optimization (backprop)
        sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                       inputData_tf_multiDimFeature_1: multiDimFeature_1_359dim, 
                                       inputData_tf_multiDimFeature_2: multiDimFeature_2_359dim, 
                                       inputData_tf_multiDimFeature_3: multiDimFeature_3_359dim, 
                                       inputData_tf_multiDimFeature_4_358dim: multiDimFeature_4_358dim, 
                                       labels_tf: labels,
                                       keep_prob_tf: keep_probability})
        if (step % display_step == 0 or step ==1):
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={  inputData_tf_oneDimFeatures: oneDimFeatures,
                                                                inputData_tf_multiDimFeature_1: multiDimFeature_1_359dim, 
                                                                inputData_tf_multiDimFeature_2: multiDimFeature_2_359dim, 
                                                                inputData_tf_multiDimFeature_3: multiDimFeature_3_359dim, 
                                                                inputData_tf_multiDimFeature_4_358dim: multiDimFeature_4_358dim, 
                                                                labels_tf: labels,
                                                                keep_prob_tf: 1.})
            if withDevEvaluation:
                acc_dev, correct_pred_dev = sess.run([accuracy, correct_pred], feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures_dev,
                                                      inputData_tf_multiDimFeature_1: jumps_dev, 
                                                      inputData_tf_multiDimFeature_2: delta_dev, 
                                                      inputData_tf_multiDimFeature_3: timeAfterJump_dev, 
                                                      inputData_tf_multiDimFeature_4_358dim: curvature_dev, 
                                                      labels_tf: labels_dev,
                                                      keep_prob_tf: 1.})
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

    ### save the model after last iteration
    if(saveModel):
        saver = tf.train.Saver()    
        saver.save(sess, path_saveModel)
        print("Model saved in file:", path_saveModel)
    
    # stream a test batch
    reader = pd.read_csv(path_testSet, header=None, sep=";", chunksize=50000, iterator=True)
    oneDimFeatures_test, delta_test, jumps_test, timeAfterJump_test, curvature_test, labels_test = next_batch_ofFeatures(reader, 5000)
    
    # Calculate accuracy for test data
    accuracy_test, correct_pred_testData = sess.run([accuracy, correct_pred], feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures_test,
                                                      inputData_tf_multiDimFeature_1: jumps_test, 
                                                      inputData_tf_multiDimFeature_2: delta_test, 
                                                      inputData_tf_multiDimFeature_3: timeAfterJump_test, 
                                                      inputData_tf_multiDimFeature_4_358dim: curvature_test, 
                                                      labels_tf: labels_test,
                                                      keep_prob_tf: 1.})
    print("Batch Accuracy:", accuracys[-1])
    print("Dev Accuracy:", accuracys_dev[-1])
    print("Testing Accuracy:", accuracy_test)

if plotCostGraph:
    usefullFunctions.plotResults(iterations, costs, accuracys, training_iters, accuracys_dev=accuracys_dev, withDevEvaluation=withDevEvaluation, withtestAccuracy=withtestAccuracy, testAccuracy=accuracy_test)

print("Number of representations in classes (test data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, labels_test.shape[0]) if (correct_pred_testData[k] == True and labels_test[k,i] == 1))
    number = sum(np.transpose(labels_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")

print(" ")

print("Number of representations in classes (dev data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, labels_dev.shape[0]) if (correct_pred_dev[k] == True and labels_dev[k,i] == 1))
    number = sum(np.transpose(labels_dev)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")
