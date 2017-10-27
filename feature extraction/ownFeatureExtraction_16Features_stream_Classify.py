#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:03:30 2017

@author: Chris
working directory is '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/feature extraction'
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from ownFeatureExtractionModels import Model_Complex_16Features
import os
import xlsxwriter
cwd = os.getcwd()
from imp import load_source
usefullFunctions = load_source("usefullFunctions", cwd + '/../usefullFunctions.py') 
featureCalculator = load_source("featureCalculator", cwd + '/../all Data/Data preparation/featureCalculator.py') 

tf.reset_default_graph()

# CSV_Name = file1Name
SourceCSV_Path = cwd + '/../all Data/Data to classify/O5887085_interpolate_english_normalizedNotional.csv'
saveModel_Path = cwd + "/trainedModels/ownFeatureExtraction_16Features_maturity.ckpt"

def loadDataFromCSVunlabled(path, normalize):
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:][:].values
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)        
    print('Data loaded')
    return datapoints

X_unlabled = loadDataFromCSVunlabled(SourceCSV_Path, False)

timeSeriesLength = 360
numberOfProfiles = X_unlabled.shape[0]

# calculate features
oneDimFeatures, delta, jumps, timeAfterJump, curvature = featureCalculator.calculate_Features_return(X_unlabled, timeSeriesLength, numberOfProfiles)

# Model specific parameters
numberWeigthsPerNeuronInFirstLayer = 20

_, n_classes = usefullFunctions.label_extraction_prams("maturity")

### define the Model as it was saved
_, _, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, inputData_tf_multiDimFeature_2, inputData_tf_multiDimFeature_3, inputData_tf_multiDimFeature_4_358dim, _, keep_prob_tf, _, _, _, softmax_tf, correct_pred = Model_Complex_16Features(n_classes, numberWeigthsPerNeuronInFirstLayer)

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, saveModel_Path)
    print("Session restored")        
    probabilities = sess.run(softmax_tf, feed_dict={inputData_tf_oneDimFeatures: oneDimFeatures,
                                                    inputData_tf_multiDimFeature_1: delta, 
                                                    inputData_tf_multiDimFeature_2: jumps, 
                                                    inputData_tf_multiDimFeature_3: timeAfterJump, 
                                                    inputData_tf_multiDimFeature_4_358dim: curvature, 
                                                    keep_prob_tf: 1.})
    print("Probabilities calculated")  

# Stack infos together
predictedClass = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((predictedClass, probabilities))

# print the results into a xlsx
workbook = xlsxwriter.Workbook(SourceCSV_Path[:-4] + '_evaluated-with-16Features-maturity-20-10-2017.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
worksheet.write_string(row, 0, "Predicted Class")
for cl in range(0, n_classes):
    worksheet.write_string(row, 1+cl, "Prob Class " + str(cl))
row = 1
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')
