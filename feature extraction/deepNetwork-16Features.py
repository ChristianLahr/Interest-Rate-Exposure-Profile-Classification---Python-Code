#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:46:34 2017

@author: Chris

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
from ownFeatureExtractionModels import Model_deepNetwork_shorter

from imp import load_source
usefullFunctions = load_source("usefullFunctions", cwd + '/../usefullFunctions.py') 


path_trainSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_train.csv'
path_devSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_dev.csv'
path_testSet = cwd + '/../all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features_test.csv'

### Parameters
learning_rate = 0.01
training_iters = 1000
batch_size = 100
display_step = 1

numberOfNeuronsInFirstLayers = 200

keep_probabilitey = 0.9

n_classes_maturity = 4
n_classes_frequency = 3
n_classes_coupon = 6

n_oneDImFeatures = 12

path_saveModel = cwd + "/trainedModels/Deep-26-10-2017.ckpt"

saveModel = False
plotCostGraph = True
withDevEvaluation = True
withtestAccuracy = True


### Be shure that the old graph is deleted
tf.reset_default_graph()  
   
    
def next_batch_ofFeatures(reader, size):
    labelNumber_maturity = 9
    labelNumber_frequency = 7
    labelNumber_coupon = 5

    stream = reader.get_chunk(size)
    # separate labels
    n_all = n_classes_maturity + n_classes_frequency + n_classes_coupon
    features            = stream.iloc[:, : 1447].values
    probs               = stream.iloc[:, 1447 : 1447 + n_all ].values
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

    oneDimFeatures                  = features[:,:12]
    # order: [maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump]
    delta                           = features[:,12:371]
    jumps                           = features[:,371:730]
    timeAfterJump                   = features[:,730:1089]
    curvature                       = features[:,1089:1447]

    probabilities_Maturity          = probs[:,:4]
    probabilities_Frequency         = probs[:,4:7]
    probabilities_Coupon            = probs[:,7:13]
            
    return oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon
 
### define the Model
init_tf, optimizer_tf, cost_tf, accuracy_Maturity_tf, accuracy_Frequency_tf, accuracy_Coupon_tf, accuracy_all_tf, inputData_tf_oneDimFeatures_tf, inputData_tf_multiDimFeature_1_tf, inputData_tf_multiDimFeature_2_tf, inputData_tf_multiDimFeature_3_tf,inputData_tf_multiDimFeature_4_358dim_tf, inputData_tf_probabilities_Maturity_tf, inputData_tf_probabilities_Frequency_tf, inputData_tf_probabilities_Coupon_tf, labels_tf_Maturity_tf, labels_tf_Frequency_tf, labels_tf_Coupon_tf, keep_prob_tf, predicted_Maturity_Class_tf, predicted_Frequency_Class_tf, predicted_Coupon_Class_tf, correct_pred_Maturity_tf, correct_pred_Frequency_tf, correct_pred_Coupon_tf, merged_tf = Model_deepNetwork_shorter(  n_oneDImFeatures, 
                                              n_classes_maturity, 
                                              n_classes_frequency, 
                                              n_classes_coupon, 
                                              numberOfNeuronsInFirstLayers, 
                                              learning_rate)

# start the stream
reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)

if withDevEvaluation:
    reader_dev = pd.read_csv(path_devSet, header=None, sep=";", chunksize=5000, iterator=True)
    oneDimFeatures_dev, delta_dev, jumps_dev, timeAfterJump_dev, curvature_dev, labels_maturity_dev, labels_frequency_dev, labels_coupon_dev, probabilities_Maturity_dev, probabilities_Frequency_dev, probabilities_Coupon_dev = next_batch_ofFeatures(reader, 5000)
    multiDimFeature_1_359dim_dev = jumps_dev
    multiDimFeature_2_359dim_dev = delta_dev
    multiDimFeature_3_359dim_dev = timeAfterJump_dev
    multiDimFeature_4_358dim_dev = curvature_dev
    accuracys_dev_all = []
    accuracys_dev_Maturity = []
    accuracys_dev_Frequency = []
    accuracys_dev_Coupon = []

costs = []
iterations = []
accuracys_all = []
accuracys_Maturity = []
accuracys_Frequency = []
accuracys_Coupon = []

### Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short', sess.graph)
    test_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short', sess.graph)
    
    sess.run(init_tf)
    print("Network variables initialized (...)")    
    step = 1

    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < (training_iters+1):
        
        # get a batch of trainings data 
        try:
            oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon = next_batch_ofFeatures(reader, batch_size)
        except StopIteration: # restart iteration if data has already passed
            print("restart reader iteration")
            reader = pd.read_csv(path_trainSet, header=None, sep=";", chunksize=batch_size, iterator=True)
            oneDimFeatures, delta, jumps, timeAfterJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon = next_batch_ofFeatures(reader, batch_size)

        multiDimFeature_1_359dim = jumps
        multiDimFeature_2_359dim = delta
        multiDimFeature_3_359dim = timeAfterJump
        multiDimFeature_4_358dim = curvature
                
        ### Run optimization op (backprop)
        
        _ = sess.run([optimizer_tf], feed_dict={inputData_tf_oneDimFeatures_tf: oneDimFeatures, 
                                                  inputData_tf_multiDimFeature_1_tf: multiDimFeature_1_359dim, 
                                                  inputData_tf_multiDimFeature_2_tf: multiDimFeature_2_359dim, 
                                                  inputData_tf_multiDimFeature_3_tf: multiDimFeature_3_359dim,
                                                  inputData_tf_multiDimFeature_4_358dim_tf: multiDimFeature_4_358dim, 
                                                  inputData_tf_probabilities_Maturity_tf: probabilities_Maturity,
                                                  inputData_tf_probabilities_Frequency_tf: probabilities_Frequency,
                                                  inputData_tf_probabilities_Coupon_tf: probabilities_Coupon,
                                                  labels_tf_Maturity_tf: labels_maturity,
                                                  labels_tf_Frequency_tf: labels_frequency,
                                                  labels_tf_Coupon_tf: labels_coupon,
                                                  keep_prob_tf: keep_probabilitey})
#        train_writer.add_summary(summary_train, step)
        
        if (step % display_step == 0 or step ==1):
            # Calculate batch loss and accuracy
            loss, acc_Maturity, acc_Frequency, acc_Coupon, acc_all = sess.run([     cost_tf,     
                                                                                    accuracy_Maturity_tf, 
                                                                                    accuracy_Frequency_tf, 
                                                                                    accuracy_Coupon_tf, 
                                                                                    accuracy_all_tf], feed_dict={inputData_tf_oneDimFeatures_tf: oneDimFeatures, 
                                                                                                                  inputData_tf_multiDimFeature_1_tf: multiDimFeature_1_359dim, 
                                                                                                                  inputData_tf_multiDimFeature_2_tf: multiDimFeature_2_359dim, 
                                                                                                                  inputData_tf_multiDimFeature_3_tf: multiDimFeature_3_359dim,
                                                                                                                  inputData_tf_multiDimFeature_4_358dim_tf: multiDimFeature_4_358dim, 
                                                                                                                  inputData_tf_probabilities_Maturity_tf: probabilities_Maturity,
                                                                                                                  inputData_tf_probabilities_Frequency_tf: probabilities_Frequency,
                                                                                                                  inputData_tf_probabilities_Coupon_tf: probabilities_Coupon,
                                                                                                                  labels_tf_Maturity_tf: labels_maturity,
                                                                                                                  labels_tf_Frequency_tf: labels_frequency,
                                                                                                                  labels_tf_Coupon_tf: labels_coupon,
                                                                                                                  keep_prob_tf: keep_probabilitey})

            if withDevEvaluation:
                acc_Maturity_dev, acc_Frequency_dev, acc_Coupon_dev, acc_all_dev = sess.run([accuracy_Maturity_tf, 
                                                                                            accuracy_Coupon_tf, 
                                                                                            accuracy_Frequency_tf, 
                                                                                            accuracy_all_tf], feed_dict={inputData_tf_oneDimFeatures_tf: oneDimFeatures_dev, 
                                                                                                                          inputData_tf_multiDimFeature_1_tf: multiDimFeature_1_359dim_dev, 
                                                                                                                          inputData_tf_multiDimFeature_2_tf: multiDimFeature_2_359dim_dev, 
                                                                                                                          inputData_tf_multiDimFeature_3_tf: multiDimFeature_3_359dim_dev,
                                                                                                                          inputData_tf_multiDimFeature_4_358dim_tf: multiDimFeature_4_358dim_dev, 
                                                                                                                          inputData_tf_probabilities_Maturity_tf: probabilities_Maturity_dev,
                                                                                                                          inputData_tf_probabilities_Frequency_tf: probabilities_Frequency_dev,
                                                                                                                          inputData_tf_probabilities_Coupon_tf: probabilities_Coupon_dev,
                                                                                                                          labels_tf_Maturity_tf: labels_maturity_dev,
                                                                                                                          labels_tf_Frequency_tf: labels_frequency_dev,
                                                                                                                          labels_tf_Coupon_tf: labels_coupon_dev,
                                                                                                                          keep_prob_tf: 1.})
                accuracys_dev_all.append(acc_all_dev)
                accuracys_dev_Maturity.append(acc_Maturity_dev)
                accuracys_dev_Frequency.append(acc_Frequency_dev)
                accuracys_dev_Coupon.append(acc_Coupon_dev)

#            test_writer.add_summary(summary_test, step)

            costs.append(loss)
            iterations.append(step*batch_size)
            accuracys_Maturity.append(acc_Maturity)
            accuracys_Frequency.append(acc_Frequency)
            accuracys_Coupon.append(acc_Coupon)
            accuracys_all.append(acc_all)

            if plotCostGraph:
                if withDevEvaluation:
                    usefullFunctions.plotResults_4Accuracies(iterations, 
                                                             costs, 
                                                             accuracys_all, 
                                                             accuracys_Maturity, 
                                                             accuracys_Frequency, 
                                                             accuracys_Coupon, 
                                                             training_iters, 
                                                             accuracys_all_dev=accuracys_dev_all, 
                                                             accuracys_Maturity_dev=accuracys_dev_Maturity, 
                                                             accuracys_Frequency_dev=accuracys_dev_Frequency, 
                                                             accuracys_Coupon_dev=accuracys_dev_Coupon, 
                                                             withDevEvaluation=withDevEvaluation)
                else:
                    usefullFunctions.plotResults_4Accuracies(iterations, 
                                                             costs, 
                                                             accuracys_all, 
                                                             accuracys_Maturity, 
                                                             accuracys_Frequency, 
                                                             accuracys_Coupon, 
                                                             training_iters)

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
    save_path = saver.save(sess, path_saveModel)
    print("Model saved in file: %s" % path_saveModel)
    
    # stream a test batch
    reader = pd.read_csv(path_testSet, header=None, sep=";", chunksize=50000, iterator=True)
    oneDimFeatures_test, delta_test, jumps_test, timeAfterJump_test, curvature_test, labels_maturity_test, labels_frequency_test, labels_coupon_test, probabilities_Maturity_test, probabilities_Frequency_test, probabilities_Coupon_test = next_batch_ofFeatures(reader, 5000)
    multiDimFeature_1_359dim_test = jumps_test
    multiDimFeature_2_359dim_test = delta_test
    multiDimFeature_3_359dim_test = timeAfterJump_test
    multiDimFeature_4_358dim_test = curvature_test

    
    # Calculate accuracy for test data
    acc_Maturity_test, acc_Frequency_test, acc_Coupon_test, acc_all_test, correct_pred_testData_Mat, correct_pred_testData_Fre, correct_pred_testData_Cou = sess.run([accuracy_Maturity_tf, 
                        accuracy_Coupon_tf, 
                        accuracy_Frequency_tf, 
                        accuracy_all_tf,
                        correct_pred_Maturity_tf, 
                        correct_pred_Frequency_tf, 
                        correct_pred_Coupon_tf], feed_dict={inputData_tf_oneDimFeatures_tf: oneDimFeatures_test, 
                                                      inputData_tf_multiDimFeature_1_tf: multiDimFeature_1_359dim_test, 
                                                      inputData_tf_multiDimFeature_2_tf: multiDimFeature_2_359dim_test, 
                                                      inputData_tf_multiDimFeature_3_tf: multiDimFeature_3_359dim_test,
                                                      inputData_tf_multiDimFeature_4_358dim_tf: multiDimFeature_4_358dim_test, 
                                                      inputData_tf_probabilities_Maturity_tf: probabilities_Maturity_test,
                                                      inputData_tf_probabilities_Frequency_tf: probabilities_Frequency_test,
                                                      inputData_tf_probabilities_Coupon_tf: probabilities_Coupon_test,
                                                      labels_tf_Maturity_tf: labels_maturity_test,
                                                      labels_tf_Frequency_tf: labels_frequency_test,
                                                      labels_tf_Coupon_tf: labels_coupon_test,
                                                      keep_prob_tf: 1.})    

    print("Batch Accuracy: all", accuracys_all[-1], "Mat", accuracys_Maturity[-1], "Freq", accuracys_Frequency[-1], "Coup:", accuracys_Coupon[-1])
    print("Dev Accuracy: all", accuracys_dev_all[-1], "Mat", accuracys_dev_Maturity[-1], "Freq", accuracys_dev_Frequency[-1], "Coup:", accuracys_dev_Coupon[-1])
    print("Testing Accuracy: all", acc_all_test, "Mat", acc_Maturity_test, "Freq", acc_Frequency_test, "Coup:", acc_Coupon_test)

if plotCostGraph:
    usefullFunctions.plotResults_4Accuracies(iterations, 
                                             costs, 
                                             accuracys_all, 
                                             accuracys_Maturity, 
                                             accuracys_Frequency, 
                                             accuracys_Coupon, 
                                             training_iters,
                                             accuracys_all_dev=accuracys_dev_all, 
                                             accuracys_Maturity_dev=accuracys_dev_Maturity, 
                                             accuracys_Frequency_dev=accuracys_dev_Frequency, 
                                             accuracys_Coupon_dev=accuracys_dev_Coupon,                                              
                                             withDevEvaluation=withDevEvaluation, 
                                             withtestAccuracy=withtestAccuracy, 
                                             testAccuracy=acc_all_test)

print("Number of representations in maturity classes (test data)")
for i in range(0, n_classes_maturity):
    correct = sum(1 for k in range(0, labels_maturity_test.shape[0]) if (correct_pred_testData_Mat[k] == True and labels_maturity_test[k,i] == 1))
    number = sum(np.transpose(labels_maturity_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")

print(" ")

print("Number of representations in frequency classes (test data)")
for i in range(0, n_classes_frequency):
    correct = sum(1 for k in range(0, labels_frequency_test.shape[0]) if (correct_pred_testData_Fre[k] == True and labels_frequency_test[k,i] == 1))
    number = sum(np.transpose(labels_frequency_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")

print(" ")

print("Number of representations in coupon classes (test data)")
for i in range(0, n_classes_coupon):
    correct = sum(1 for k in range(0, labels_coupon_test.shape[0]) if (correct_pred_testData_Cou[k] == True and labels_coupon_test[k,i] == 1))
    number = sum(np.transpose(labels_coupon_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")

