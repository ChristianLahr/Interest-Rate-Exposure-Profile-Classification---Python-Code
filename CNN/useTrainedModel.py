#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:01:53 2017

@author: Chris
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from conv1d_TensorFlow_CNN_360_stream_v02 import loadDataFromCSVunlabled, Model

tf.reset_default_graph()

# CSV_Name = file1Name
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/O5887085_interpolate_english_normalizedNotional.csv'

X_unlabled = loadDataFromCSVunlabled(CSV_Name, False, False)

init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, pred2, correct_pred = Model()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)    
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-360-stream-Coupon-test.ckpt")
    print("Session restored")    
    logits = sess.run(pred, feed_dict={   x: X_unlabled,
                                            keep_prob: 1.})
    print("Probabilities calculated")  
    logits_scaled = logits / 100000 # das scaling an dieser Stelle verändert die Wkeiten stark, aber nicht die Reihenfolge
    probabilities = tf.nn.softmax(logits_scaled).eval()


# Stack infos together
probabilities_argmax = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((probabilities_argmax, probabilities))

# print the probabilities into a xlsx
import xlsxwriter
workbook = xlsxwriter.Workbook(CSV_Name[:-4] + '_evaluated-test.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')

