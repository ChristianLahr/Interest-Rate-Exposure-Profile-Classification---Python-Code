#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:23:34 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_deepNetwork_shorter_72classes
from featureCalculator_np import featureCalculator_np
from usefullFunctions import loadAllLabelsFromCSV_72classes, randomizeRowsOfArraysInList
from useTrainedModel_fct import useTrainedModel


### Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 1000
display_step = 10

numberWeigthsPerNeuronInFirstLayer = 600 # aktuell sind die berechneten Modelle mit 200 ausgewertet und wollen diesen Wert auch hier haben
dataNeuLadenUndFeaturesNeuBerechnen = False
useTrainedModels = False

dropout = 0.9

labelNumber_maturity = 9
labelNumber_frequency = 7
labelNumber_coupon = 5

n_classes_maturity = 4
n_classes_frequency = 3
n_classes_coupon = 6
## labels:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

### Be shure that the old graph is desleted
tf.reset_default_graph()

### Load Exposure Profiles
if(dataNeuLadenUndFeaturesNeuBerechnen):
    file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Feature Extraction Test Data/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
    data_startOrder, labels_72classes_startOrder = loadAllLabelsFromCSV_72classes(file1Name)
    data_startOrder = data_startOrder[:, :]
    labels_72classes_startOrder = labels_72classes_startOrder[:,:]

    ### clear the data from zero lines
    zeroLine = np.zeros(360)
    badIndexes = []
    for index, item in enumerate(data_startOrder):
    #    if(index==0): print(item)
        if(np.array_equal(item, zeroLine)): badIndexes.append(index)
    for i in reversed(badIndexes):
        data_startOrder = np.delete(data_startOrder, i, 0)
        labels_72classes_startOrder = np.delete(labels_72classes_startOrder, i, 0)

    timeSeriesLength = data_startOrder.shape[1]
    numberOfProfiles = data_startOrder.shape[0]

### Features Berechnen
    maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump = featureCalculator_np(data_startOrder, timeSeriesLength, numberOfProfiles)
   
### alway start in same order! Otherwise labels and data is perhaps independently randomized
data = data_startOrder
labels_72classes = labels_72classes_startOrder

### network input for trained models
networkInput = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump, jumps]
networkInput_length = len(networkInput)
networkInput_length_oneDImFeatures = networkInput_length - 1
    
print("use", networkInput_length, "features as network input for pre classification")

if(useTrainedModels):
    ### load Data from trained models as extra input for deeper model
    print("Calculate trained model outputs to get more information for the deep network")
    
    name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Maturity.ckpt'
    probabilities_startOrder_Maturity = useTrainedModel(networkInput, name_SavesModel, labelNumber_maturity, n_classes_maturity, numberWeigthsPerNeuronInFirstLayer)
    
    name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Frequency.ckpt'
    probabilities_startOrder_Frequency = useTrainedModel(networkInput, name_SavesModel, labelNumber_frequency, n_classes_frequency, numberWeigthsPerNeuronInFirstLayer)
    
    name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Coupon.ckpt'
    probabilities_startOrder_Coupon = useTrainedModel(networkInput, name_SavesModel, labelNumber_coupon, n_classes_coupon, numberWeigthsPerNeuronInFirstLayer)
    
    print("Additional inputs calculated")

probabilities_Maturity = probabilities_startOrder_Maturity  
probabilities_Frequency = probabilities_startOrder_Frequency
probabilities_Coupon = probabilities_startOrder_Coupon
    
### network input for deep network incl. labels
### need all features as list to randomize the order in the same way for every featurearray, no matter which dimension it has
networkInput = [maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist, numberOfJumpsRelativeToLength, maximalJump, jumps, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon]

### define the Model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_72classes_tf, keep_prob_tf, weights, biases, prediction, merged = Model_deepNetwork_shorter_72classes(  networkInput_length_oneDImFeatures, 
                                              n_classes_maturity, 
                                              n_classes_frequency, 
                                              n_classes_coupon, 
                                              numberWeigthsPerNeuronInFirstLayer, 
                                              learning_rate)
               
### take mini-batch of trainings data
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

### randomize data before traing start
networkInput, labels_72classes = randomizeRowsOfArraysInList(networkInput, labels_72classes) # labels are in network input
networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDImFeatures]
networkInput_prob_Maturity = networkInput[networkInput_length_oneDImFeatures + 1]
networkInput_prob_Frequency = networkInput[networkInput_length_oneDImFeatures + 2]
networkInput_prob_Coupon = networkInput[networkInput_length_oneDImFeatures + 3]

train_length = len(data_startOrder)

### Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short 72 classes', sess.graph)
    test_writer = tf.summary.FileWriter('/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/TensorBoardData/deep network short 72 classes', sess.graph)
    
    sess.run(init)
    print("Network variables initialized (", numberWeigthsPerNeuronInFirstLayer * networkInput_length_oneDImFeatures + 359 * 5 * numberWeigthsPerNeuronInFirstLayer + 13 * 5 * numberWeigthsPerNeuronInFirstLayer + 19,5*numberWeigthsPerNeuronInFirstLayer * 19,5*numberWeigthsPerNeuronInFirstLayer / 2 + 19,5*numberWeigthsPerNeuronInFirstLayer /2 * 19,5*numberWeigthsPerNeuronInFirstLayer/4 + 19,5*numberWeigthsPerNeuronInFirstLayer/4 * 19,5*numberWeigthsPerNeuronInFirstLayer/10 * 3 + 19,5*numberWeigthsPerNeuronInFirstLayer/10 * n_classes_maturity + 19,5*numberWeigthsPerNeuronInFirstLayer / 10 * n_classes_frequency + 19,5*numberWeigthsPerNeuronInFirstLayer / 10 * n_classes_coupon, ")")    
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
            networkInput, labels_72classes = randomizeRowsOfArraysInList(networkInput, labels_72classes)
            networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
            networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
            networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDImFeatures]
            networkInput_prob_Maturity = networkInput[networkInput_length_oneDImFeatures + 1]
            networkInput_prob_Frequency = networkInput[networkInput_length_oneDImFeatures + 2]
            networkInput_prob_Coupon = networkInput[networkInput_length_oneDImFeatures + 3]

            print('round ', r+1, "   ", step)
        batch_x_1 = next_batch(networkInput_oneDimFeatures, batch_size, batch_number)
        batch_x_2 = next_batch(networkInput_multiDimFeature_1, batch_size, batch_number)
        batch_x_3 = next_batch(networkInput_prob_Maturity, batch_size, batch_number)
        batch_x_4 = next_batch(networkInput_prob_Frequency, batch_size, batch_number)
        batch_x_5 = next_batch(networkInput_prob_Coupon, batch_size, batch_number)
        batch_y = next_batch(labels_72classes, batch_size, batch_number)

        # weight normalization before every update                
        for index, key in enumerate(weights):        
            weights[list(weights.keys())[index]].eval()
            biases[list(biases.keys())[index]].eval()
                
        batch_number += 1
        ### Run optimization op (backprop)
        summary_train, _ = sess.run([merged, optimizer], feed_dict={ inputData_tf_oneDimFeatures: batch_x_1,
                                        inputData_tf_multiDimFeature_jumps: batch_x_2,
                                        inputData_tf_probabilities_Maturity: batch_x_3,
                                        inputData_tf_probabilities_Frequency: batch_x_4,
                                        inputData_tf_probabilities_Coupon: batch_x_5,
                                        labels_72classes_tf: batch_y,
                                        keep_prob_tf: dropout})
        train_writer.add_summary(summary_train, i)
        
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary_test, loss, acc = sess.run([ merged, 
                                                                                cost,     
                                                                                accuracy], feed_dict={  inputData_tf_oneDimFeatures: batch_x_1,
                                                                                                        inputData_tf_multiDimFeature_jumps: batch_x_2, 
                                                                                                        inputData_tf_probabilities_Maturity: batch_x_3,
                                                                                                        inputData_tf_probabilities_Frequency: batch_x_4,
                                                                                                        inputData_tf_probabilities_Coupon: batch_x_5,
                                                                                                        labels_72classes_tf: batch_y,
                                                                                                        keep_prob_tf: 1.})

            test_writer.add_summary(summary_test, step)

            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Accuracy: " + \
                  "{:.5f}".format(acc))
        step += 1

        ### save the model after last iteration
        if(step * batch_size > training_iters - batch_size):            
            saver = tf.train.Saver()    
            save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_DeepNetwork_shorter_72classes_test.ckpt")
            print("Model saved in file: %s" % save_path)
            
    print("Training Finished!")
    
    # Calculate accuracy for test data
    overallTest = sess.run(accuracy, feed_dict={inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                                         inputData_tf_multiDimFeature_jumps: networkInput_multiDimFeature_1,
                                                         inputData_tf_probabilities_Maturity: networkInput_prob_Maturity,
                                                         inputData_tf_probabilities_Frequency: networkInput_prob_Frequency,
                                                         inputData_tf_probabilities_Coupon: networkInput_prob_Coupon,
                                                         labels_72classes_tf: labels_72classes,
                                                         keep_prob_tf: 1.})
    
    print("Accuracy:", overallTest)

### save the run params in a dict
newEntry = {"test_accuracy" : overallTest,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "dropout" : dropout,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "numberWeigthsPerNeuronInFirstLayer": numberWeigthsPerNeuronInFirstLayer,
            "Model": 'Model_DeepNetwork_shorter',
            "comment" : 'inkl Inputs aus gespeicherten Modellen'}  
results.append(newEntry)



# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
