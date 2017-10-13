#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:51:19 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_Complex_16Features
import pandas as pd

fileName_Features = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'

### Parameters
learning_rate = 0.1
training_iters = 50000
# batch size should not be bigger than available datasets (because than every batch = all data --> meaningless)
batch_size = 500
display_step = 1

numberWeigthsPerNeuronInFirstLayer = 200
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

n_oneDImFeatures = 12

### define the Model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, labels_tf, keep_prob_tf, weights, biases, prediction, correct_pred = Model_Complex_16Features(n_oneDImFeatures, n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate)

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
 
        oneDimFeatures           = [maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump]
        oneDimFeatures           = np.transpose(oneDimFeatures)
        multiDimFeature_1_359dim = jumps
        multiDimFeature_2_359dim = delta
        multiDimFeature_3_359dim = timeAfterJump
        multiDimFeature_2_358dim = curvature

        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                       inputData_tf_multiDimFeature_1: multiDimFeature_1_359dim, 
                                       inputData_tf_multiDimFeature_2: multiDimFeature_2_359dim, 
                                       inputData_tf_multiDimFeature_3: multiDimFeature_3_359dim, 
                                       inputData_tf_multiDimFeature_4_358dim: multiDimFeature_2_358dim, 
                                       labels_tf: labels,
                                       keep_prob_tf: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={  inputData_tf_oneDimFeatures: oneDimFeatures,
                                                                inputData_tf_multiDimFeature_1: multiDimFeature_1_359dim, 
                                                                inputData_tf_multiDimFeature_2: multiDimFeature_2_359dim, 
                                                                inputData_tf_multiDimFeature_3: multiDimFeature_3_359dim, 
                                                                inputData_tf_multiDimFeature_4_358dim: multiDimFeature_2_358dim, 
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
    
    biggerBatchSize = 50000
    reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=biggerBatchSize, iterator=True)
    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels = next_batch_ofFeatures(reader, biggerBatchSize)
    oneDimFeatures           = [maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump]
    oneDimFeatures           = np.transpose(oneDimFeatures)
    multiDimFeature_1_359dim = jumps
    multiDimFeature_2_359dim = delta
    multiDimFeature_3_359dim = timeAfterJump
    multiDimFeature_2_358dim = curvature

    # Calculate accuracy for test data
    overallTest, correct_predictedOfOverallTest = sess.run([accuracy, correct_pred], feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                                      inputData_tf_multiDimFeature_1: multiDimFeature_1_359dim, 
                                                      inputData_tf_multiDimFeature_2: multiDimFeature_2_359dim, 
                                                      inputData_tf_multiDimFeature_3: multiDimFeature_3_359dim, 
                                                      inputData_tf_multiDimFeature_4_358dim: multiDimFeature_2_358dim, 
                                                      labels_tf: labels,
                                                      keep_prob_tf: 1.})
    print("Testing Accuracy:", overallTest)
    
print("Number of representations in classes (all data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, labels.shape[0]) if (correct_predictedOfOverallTest[k] == True and labels[k,i] == 1))
    number = sum(np.transpose(labels)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")    

# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
