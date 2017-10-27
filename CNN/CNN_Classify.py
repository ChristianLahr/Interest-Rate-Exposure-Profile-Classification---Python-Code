#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:01:53 2017

@author: Chris

working directory is '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/CNN'
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from CNN_Model import Model
import os
import xlsxwriter
cwd = os.getcwd()
from imp import load_source
usefullFunctions = load_source("usefullFunctions", cwd + '/../usefullFunctions.py') 

tf.reset_default_graph()

# CSV_Name = file1Name
SourceCSV_Path = cwd + '/../all Data/Data to classify/O5887085_interpolate_english_normalizedNotional.csv'
saveModel_Path = cwd + "/trainedModels/CNN-16-10-2017.ckpt"

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

conv1Kernal_size = 15
conv1Feature_Number = 16
conv2Kernal_size = 30
conv2Feature_Number = 32

_, n_classes = usefullFunctions.label_extraction_prams("maturity")

### define the Model as it was saved
init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, CNN_output_tf, softmax_tf, correct_pred = Model(n_classes, conv1Feature_Number, conv2Feature_Number, conv1Kernal_size, conv2Kernal_size)

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, saveModel_Path)
    print("Session restored")    
    probabilities = sess.run(softmax_tf, feed_dict={x: X_unlabled, keep_prob: 1.})
    print("Probabilities calculated")  

# Stack infos together
predictedClass = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((predictedClass, probabilities))

# print the results into a xlsx
workbook = xlsxwriter.Workbook(SourceCSV_Path[:-4] + '_evaluated-with-CNN-16-10-2017.xlsx')
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
