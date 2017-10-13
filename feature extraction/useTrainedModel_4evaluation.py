#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:01:53 2017

@author: Chris
"""
import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_Complex_multiDimFeatures, Model_deepNetwork_shorter
from featureCalculator_np import featureCalculator_np
from usefullFunctions import loadDataFromCSVunlabeled
from useTrainedModel_fct import useTrainedModel, useTrainedModel_DeepNetwork

###  Parameters (need to be the same as used to train the model!!)
numberWeigthsPerNeuronInFirstLayer = 100
learning_rate = 0.01
lableNumber = 5
n_classes = 6
dropout = 0.9
featuresNeuBerechnen = True

### Be shure that the old graph is deleted
tf.reset_default_graph()

### Load Exposure Profiles
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/O5887085_interpolate_english_normalizedNotional.csv'
data_e = loadDataFromCSVunlabeled(CSV_Name, False, False)

timeSeriesLength_e = data_e.shape[1]
numberOfProfiles_e = data_e.shape[0]

### calculate the features and store them in the rigth way
if(featuresNeuBerechnen):
   maximum_e, maximumPosition_e, entryPoint_e, deltaMaxFist_e, delta_e, jumps_e, timeAfterJump_e, numberOfZeros_e, maximalDistanceToTheNextDrop_e, mean_e, variance_e, numberOfJumps_e, averageJump_e, numberOfJumpsRelativeToLength_e, maximalJump_e = featureCalculator_np(data_e, timeSeriesLength_e, numberOfProfiles_e)
    
networkInput = [maximum_e, maximumPosition_e, entryPoint_e, numberOfZeros_e, mean_e, variance_e, maximalDistanceToTheNextDrop_e, deltaMaxFist_e, numberOfJumpsRelativeToLength_e, maximalJump_e, jumps_e]
networkInput_length = len(networkInput)
networkInput_length_oneDImFeatures = networkInput_length - 1

networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDImFeatures]

### load the required model
init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, lables_tf, keep_prob_tf, weights, biases, prediction = Model_Complex_multiDimFeatures(networkInput_length_oneDImFeatures, n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate)
# init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, pred, initialValues_wc1_values = Model()

### Launch the graph
with tf.Session() as sess:
    sess.run(init)    
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_MultiDimInput_frequency.ckpt")
    print("Session restored")
    logits = sess.run(prediction, feed_dict={inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                       inputData_tf_multiDimFeature_1: networkInput_multiDimFeature_1,
                                       keep_prob_tf: 1.})
    print("Probabilities calculated")  
    probabilities = tf.nn.softmax(logits).eval()

### Stack infos together
probabilities_argmax = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((probabilities_argmax, probabilities))

### Print the probabilities into a xlsx
import xlsxwriter
workbook = xlsxwriter.Workbook(CSV_Name[:-4] + '_evaluated.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')


