#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:23:34 2017

@author: Chris

use calculated features: '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'
use trained models:
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from ownFeatureExtractionModels import Model_deepNetwork, Model_deepNetwork_shorter

fileName_Features = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'

### Parameters
learning_rate = 0.01
training_iters = 20000
batch_size = 4000
display_step = 1

numberOfNeuronsInFirstLayers = 200

dropout = 0.9

n_classes_maturity = 4
n_classes_frequency = 3
n_classes_coupon = 6
## labels:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

### Be shure that the old graph is deleted
tf.reset_default_graph()  
   
def next_batch_ofFeatures(reader, size):
    labelNumber_maturity = 9
    labelNumber_frequency = 7
    labelNumber_coupon = 5

    stream = reader.get_chunk(size)
    # separate labels
    n_all = n_classes_maturity + n_classes_frequency + n_classes_coupon
    n_feature = stream.shape[1] - 10 -  n_all
    features            = stream.iloc[:, : n_feature].values
    probs               = stream.iloc[:, n_feature : n_feature + n_all ].values
    labels_maturity     = stream.iloc[:, stream.shape[1] - labelNumber_maturity : stream.shape[1] - (labelNumber_maturity-1) ].values
    labels_frequency    = stream.iloc[:, stream.shape[1] - labelNumber_frequency : stream.shape[1] - (labelNumber_frequency-1) ].values
    labels_coupon       = stream.iloc[:, stream.shape[1] - labelNumber_coupon : stream.shape[1] - (labelNumber_coupon-1) ].values
                                      
    # create one hot vector for labels
    def oneHot(length, n_classes, lab):
        a = np.zeros((features.shape[0], n_classes)) 
        for i, number in enumerate(lab):
            a[i][ int(number[0]) ] = 1        
        return a
            
    labels_maturity = oneHot(size, n_classes_maturity, labels_maturity)
    labels_frequency = oneHot(size, n_classes_frequency, labels_frequency)
    labels_coupon = oneHot(size, n_classes_coupon, labels_coupon)

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

    probabilities_Maturity          = probs[:,:4]
    probabilities_Frequency         = probs[:,4:7]
    probabilities_Coupon            = probs[:,7:13]

    return maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon
 
n_oneDImFeatures = 11

### define the Model
init, optimizer, cost, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf_Maturity, labels_tf_Frequency, labels_tf_Coupon, keep_prob_tf, weights, biases, _, _, _, _, _, _, merged = Model_deepNetwork_shorter(  n_oneDImFeatures, 
                                              n_classes_maturity, 
                                              n_classes_frequency, 
                                              n_classes_coupon, 
                                              numberOfNeuronsInFirstLayers, 
                                              learning_rate)

# start the stream
reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=batch_size, iterator=True)

### Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short', sess.graph)
    test_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short', sess.graph)
    
    sess.run(init)
    print("Network variables initialized (...)")    
    step = 1

# calculate features with tf: (does not work at the moment)
#    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = sess.run(optimizer, feed_dict={inputData_tf_oneDimFeatures: batch_x_1})

    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < (training_iters+1):
        
        # get a batch of trainings data 
        try:
            maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon = next_batch_ofFeatures(reader, batch_size)
        except StopIteration: # restart iteration if data has already passed
            print("restart reader iteration")
            reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=batch_size, iterator=True)
            maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon = next_batch_ofFeatures(reader, batch_size)

        oneDimFeatures      = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump, averageJump]
        oneDimFeatures      = np.transpose(oneDimFeatures)
        multiDimFeatures    = jumps

        # weight normalization before every update                
        for index, key in enumerate(weights):        
            weights[list(weights.keys())[index]].eval()
            biases[list(biases.keys())[index]].eval()
                
        ### Run optimization op (backprop)
        summary_train, _ = sess.run([merged, optimizer], feed_dict={ inputData_tf_oneDimFeatures: oneDimFeatures,
                                        inputData_tf_multiDimFeature_jumps: multiDimFeatures,
                                        inputData_tf_probabilities_Maturity: probabilities_Maturity,
                                        inputData_tf_probabilities_Frequency: probabilities_Frequency,
                                        inputData_tf_probabilities_Coupon: probabilities_Coupon,
                                        labels_tf_Maturity: labels_maturity,
                                        labels_tf_Frequency: labels_frequency,
                                        labels_tf_Coupon: labels_coupon,
                                        keep_prob_tf: dropout})
        train_writer.add_summary(summary_train, step)
        
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary_test, loss, acc_Maturity, acc_Frequency, acc_Coupon, acc_all = sess.run([ merged, 
                                                                                cost,     
                                                                                accuracy_Maturity, 
                                                                                accuracy_Frequency, 
                                                                                accuracy_Coupon, 
                                                                                accuracy_all], feed_dict={  inputData_tf_oneDimFeatures: oneDimFeatures,
                                                                                                            inputData_tf_multiDimFeature_jumps: multiDimFeatures, 
                                                                                                            inputData_tf_probabilities_Maturity: probabilities_Maturity,
                                                                                                            inputData_tf_probabilities_Frequency: probabilities_Frequency,
                                                                                                            inputData_tf_probabilities_Coupon: probabilities_Coupon,
                                                                                                            labels_tf_Maturity: labels_maturity,
                                                                                                            labels_tf_Frequency: labels_frequency,
                                                                                                            labels_tf_Coupon: labels_coupon,
                                                                                                            keep_prob_tf: 1.})

            test_writer.add_summary(summary_test, step)

            print("Iter " + str(step * batch_size) + ", Batch Loss= " + \
                  "{:.6f}".format(loss) + ", Mat " + \
                  "{:.5f}".format(acc_Maturity) + ", Fre " + \
                  "{:.5f}".format(acc_Frequency) + ", Cou " + \
                  "{:.5f}".format(acc_Coupon) + ", All " + \
                  "{:.5f}".format(acc_all))
        step += 1            

    print("Training Finished!")

    ### save the model after last iteration
    saver = tf.train.Saver()    
    save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_DeepNetwork_shorter_test2.ckpt")
    print("Model saved in file: %s" % save_path)
    
    # load test data
    biggerBatchSize = 10000
    reader = pd.read_csv(fileName_Features, header=None, sep=";", chunksize=biggerBatchSize, iterator=True)
    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon = next_batch_ofFeatures(reader, biggerBatchSize)
    oneDimFeatures      = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump, averageJump]
    oneDimFeatures      = np.transpose(oneDimFeatures)
    multiDimFeatures    = jumps
    
    # Calculate accuracy for test data
    overallTest_Maturity, overallTest_Frequency, overallTest_Coupon, overallTest = sess.run([ accuracy_Maturity, 
                                                                        accuracy_Frequency, 
                                                                        accuracy_Coupon, 
                                                                        accuracy_all], feed_dict={  inputData_tf_oneDimFeatures: oneDimFeatures,
                                                                                                    inputData_tf_multiDimFeature_jumps: multiDimFeatures, 
                                                                                                    inputData_tf_probabilities_Maturity: probabilities_Maturity,
                                                                                                    inputData_tf_probabilities_Frequency: probabilities_Frequency,
                                                                                                    inputData_tf_probabilities_Coupon: probabilities_Coupon,
                                                                                                    labels_tf_Maturity: labels_maturity,
                                                                                                    labels_tf_Frequency: labels_frequency,
                                                                                                    labels_tf_Coupon: labels_coupon,
                                                                                                    keep_prob_tf: 1.})
    
    print("Maturity Accuracy:", overallTest_Maturity)
    print("Frequency Accuracy:", overallTest_Frequency)
    print("Coupon Accuracy:", overallTest_Coupon)
    print("Overall Accuracy:", overallTest)





# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
