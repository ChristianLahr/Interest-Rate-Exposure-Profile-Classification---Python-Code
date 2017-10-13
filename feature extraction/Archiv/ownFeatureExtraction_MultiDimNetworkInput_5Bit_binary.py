#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:54:53 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import  Model_Complex_multiDimFeatures_5Bit_binary
from featureCalculator_np import featureCalculator_np
from usefullFunctions import loadDataFromCSVrealNumber, randomizeRowsOfArraysInList

### Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 1000
display_step = 10

numberWeigthsPerNeuronInFirstLayer = 200
# be careful: if features are not calculated again but data is loaded again to get a different label, we habe the problem that data and features are in a different row!!! (features got randomized but data not)
featuresNeuBerechnen = False
dataNeuLaden = False
saveModel = False

labelNumber = 9
n_classes = 4
dropout = 0.9
## labels:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

### Be shure that the old graph is desleted
tf.reset_default_graph()

### Load Exposure Profiles
if(dataNeuLaden):
    file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Feature Extraction Test Data/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
    data_startOder, labels_realNumber_startOrder = loadDataFromCSVrealNumber(file1Name, labelNumber+1) # normalize?, randomize ? 
    data_startOder = data_startOder[:, :]
    labels_realNumber_startOrder = labels_realNumber_startOrder[:,:]

    ### clear the data from zero lines
    zeroLine = np.zeros(360)
    badIndexes = []
    for index, item in enumerate(data_startOder):
    #    if(index==0): print(item)
        if(np.array_equal(item, zeroLine)): badIndexes.append(index)
    for i in reversed(badIndexes):
        data_startOder = np.delete(data_startOder, i, 0)
        labels_realNumber_startOrder = np.delete(labels_realNumber_startOrder, i, 0)

    timeSeriesLength = data_startOder.shape[1]
    numberOfProfiles = data_startOder.shape[0]

### Features Berechnen
if(featuresNeuBerechnen):
    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = featureCalculator_np(data_startOder, timeSeriesLength, numberOfProfiles)
    
### alway start in same order! Otherwise labels and data is perhaps independently randomized    
labels_realNumber = labels_realNumber_startOrder

### calculate features with tf: (does not work at the moment) (should be a permformance boost)
# maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = featureCalculation(timeSeriesLength, numberOfProfiles)

### need all features as list to randomize the order in the same way for every featurearray, no matter which dimension it has
networkInput = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump, jumps]
networkInput_length = len(networkInput)
networkInput_length_oneDImFeatures = networkInput_length - 1

print("use", networkInput_length, "features as network input")

### define the Model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, labels_realNumber_tf, keep_prob_tf, weights, biases, predictionsAsNumber, bit1, tensorInBitRepresentation =  Model_Complex_multiDimFeatures_5Bit_binary(networkInput_length_oneDImFeatures, numberWeigthsPerNeuronInFirstLayer, learning_rate)

### take mini-batch of trainings data
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

### randomize data before traing start
networkInput, labels_realNumber = randomizeRowsOfArraysInList(networkInput, labels_realNumber)
networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDImFeatures]

train_length = len(data_startOder)

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (", numberWeigthsPerNeuronInFirstLayer * networkInput_length_oneDImFeatures + numberWeigthsPerNeuronInFirstLayer * numberWeigthsPerNeuronInFirstLayer/2 + numberWeigthsPerNeuronInFirstLayer/2 * numberWeigthsPerNeuronInFirstLayer/4 + numberWeigthsPerNeuronInFirstLayer/4 * numberWeigthsPerNeuronInFirstLayer/10 + numberWeigthsPerNeuronInFirstLayer/10 * n_classes, ")")    
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)

# calculate features with tf: (does not work at the moment)
#    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: batch_x_1})

    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        ### start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0
            r += 1
            # new order of rows of data for next training run
            networkInput, labels_realNumber = randomizeRowsOfArraysInList(networkInput, labels_realNumber)
            networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
            networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
            networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDImFeatures]
            print('round ', r+1, "   ", step)
        batch_x_1 = next_batch(networkInput_oneDimFeatures, batch_size, batch_number)
        batch_x_2 = next_batch(networkInput_multiDimFeature_1, batch_size, batch_number)
        batch_y = next_batch(labels_realNumber, batch_size, batch_number)
        batch_number += 1
        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: batch_x_1,
                                       inputData_tf_multiDimFeature_1: batch_x_2,
                                       labels_realNumber_tf: batch_y,
                                       keep_prob_tf: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, bit1_torpint, tensorInBitRepresentation_torpint = sess.run([cost, accuracy, bit1, tensorInBitRepresentation], feed_dict={  inputData_tf_oneDimFeatures: batch_x_1,
                                                                inputData_tf_multiDimFeature_1: batch_x_2, 
                                                                labels_realNumber_tf: batch_y,
                                                                keep_prob_tf: 1.})
            print("hier#####  ", bit1_torpint, tensorInBitRepresentation_torpint )
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
        ### save the model after last iteration
        if(saveModel):
            if(step * batch_size > training_iters - batch_size):            
                saver = tf.train.Saver()    
                save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_MultiDimInput_binary_Maturity_test.ckpt")
                print("Model saved in file: %s" % save_path)
            
    print("Training Finished!")
    
    # Calculate accuracy for test data
    overallTest = float(sess.run(accuracy, feed_dict={inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                                      inputData_tf_multiDimFeature_1: networkInput_multiDimFeature_1,
                                                      labels_realNumber_tf: labels_realNumber,
                                                      keep_prob_tf: 1.}))
    print("Testing Accuracy:", overallTest)
    
### save the run params in a dict
newEntry = {"test_accuracy" : overallTest,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "dropout" : dropout,
            "n_classes" : n_classes,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "numberWeigthsPerNeuronInFirstLayer": numberWeigthsPerNeuronInFirstLayer,
            "labelNumber": labelNumber,
            "Model": 'Model_together',
            "comment" : 'train model coupon for deep network'}  
results.append(newEntry)


# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
