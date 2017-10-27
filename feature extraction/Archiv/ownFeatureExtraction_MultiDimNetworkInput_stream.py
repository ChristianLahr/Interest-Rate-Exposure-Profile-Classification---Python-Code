#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:54:53 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_Complex_multiDimFeatures
import pandas as pd

fileName_Features = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'

### Parameters
learning_rate = 0.01
training_iters = 10000
# batch size should not be very bigger than available datasets (because than every batch = all data --> meaningless)
batch_size = 1000
display_step = 1

numberWeigthsPerNeuronInFirstLayer = 200
# be careful: if features are not calculated again but data is loaded again to get a different label, we habe the problem that data and features are in a different row!!! (features got randomized but data not)
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

    maximum                         = features[:,0]
    maximumPosition                 = features[:,1]
    entryPoint                      = features[:,2]
    deltaMaxFist                    = features[:,3]
    delta                           = features[:,4:363]
    jumps                           = features[:,363:722]
    timeAfterJump                   = features[:,722:1081]
    numberOfZeros                   = features[:,1081]
    maximalDistanceToTheNextDrop    = features[:,1082]
    mean                            = features[:,1083]
    variance                        = features[:,1084]
    numberOfJumps                   = features[:,1085]
    averageJump                     = features[:,1086]
    numberOfJumpsRelativeToLength   = features[:,1087]
    maximalJump                     = features[:,1088]
    curvature                       = features[:,1089:1447]

    return maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels

n_oneDImFeatures = 10

### define the Model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, labels_tf, keep_prob_tf, weights, biases, prediction = Model_Complex_multiDimFeatures(n_oneDImFeatures, n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate)

# start the stream
reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=batch_size, iterator=True)

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (...)")    
    step = 1

# calculate features with tf: (does not work at the moment)
#    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: batch_x_1})

    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        try:
            maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels = next_batch_ofFeatures(reader, batch_size)
        except StopIteration: # restart iteration if data has already passed
            print("restart reader iteration")
            reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=batch_size, iterator=True)
            maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels = next_batch_ofFeatures(reader, batch_size)

        oneDimFeatures      = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump]
        oneDimFeatures      = np.transpose(oneDimFeatures)
        multiDimFeatures    = jumps

        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                       inputData_tf_multiDimFeature_1: multiDimFeatures,
                                       labels_tf: labels,
                                       keep_prob_tf: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={  inputData_tf_oneDimFeatures: oneDimFeatures,
                                                                inputData_tf_multiDimFeature_1: multiDimFeatures, 
                                                                labels_tf: labels,
                                                                keep_prob_tf: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
                    
    print("Training Finished!")

    ### save the model after last iteration
    if(saveModel):
        saver = tf.train.Saver()    
        save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_MultiDimInput_test.ckpt")
        print("Model saved in file: %s" % save_path)
    
    biggerBatchSize = 10000
    reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=biggerBatchSize, iterator=True)
    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels = next_batch_ofFeatures(reader, biggerBatchSize)
    oneDimFeatures      = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump]
    oneDimFeatures      = np.transpose(oneDimFeatures)
    multiDimFeatures    = jumps

    # Calculate accuracy for test data
    overallTest = float(sess.run(accuracy, feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                                      inputData_tf_multiDimFeature_1: multiDimFeatures,
                                                      labels_tf: labels,
                                                      keep_prob_tf: 1.}))
    print("Testing Accuracy:", overallTest)
    

# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
