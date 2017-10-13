#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:42:39 2017

@author: Chris

A Convolutional Network implementation using TensorFlow library.
"""

# first define results, but only one time. Than all the results are archived there
# results=[{}]
#%%

def randomizeRows(VectData, VectLabel):
    # brings the rows of both vectors in the same new random order
    # both vectors need same length
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return tempData, tempLabel

def randomizeRowsUnlabled(VectData):
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    return tempData
    
def loadDataFromCSV(path, normalize, randomize, lableNumber): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    ###### select here the lables to be analyzed
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    lables = completeData.iloc[:, completeData.shape[1] - lableNumber : completeData.shape[1] - (lableNumber-1) ].values
    ###### select here the lables to be analyzed

    a = np.zeros((datapoints.shape[0], n_classes)) 
    for i, number in enumerate(lables):
        a[i][ int(number[0]) ] = 1        

    lables_Matrix = a
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, lables_rand_list = randomizeRows(datapoints, lables_Matrix)
        datapoints_rand = np.float32(datapoints_rand_list)
        lables_rand = np.float32(lables_rand_list)
    else: 
        datapoints_rand = np.float32(datapoints)
        lables_rand = np.float32(lables_Matrix)
    print('Data loaded')
    return datapoints_rand, lables_rand

def loadDataFromCSVunlabled(path, normalize, randomize): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:][:].values
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, lables_rand_list = randomizeRowsUnlabled(datapoints)
        datapoints_rand = np.float32(datapoints_rand_list)
    else: 
        datapoints_rand = np.float32(datapoints)
        
    print('Data loaded')
    return datapoints_rand
    
def Model():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, timeSeriesLength])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    
    # Create some wrappers for simplicity
    def conv1d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        # see padding explanation here http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    
    def maxpool1d(x, k=2):   
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                              padding='SAME')
    
    
    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, timeSeriesLength, 1])
    
        # Convolution Layer
        conv1 = conv1d(x, weights['wc1'], biases['bc1'], strides = 1 )
        
        # rectivied linear unit (non-linear activation)
        conv1 = tf.nn.relu(conv1)
        
        # Max Pooling (down-sampling)
        conv1 = tf.reshape(conv1, shape=[-1, timeSeriesLength, 1, conv1Feature_Number])
        maxp1 = maxpool1d(conv1, k=2)
    #    maxp1 = tf.nn.local_response_normalization(maxp1)
        maxp1 = tf.reshape(maxp1, shape=[-1, int(timeSeriesLength/2), conv1Feature_Number])
            
        # Fully connected layer 1
        # Reshape maxp2 output to fit fully connected layer input
        fc1 = tf.reshape(maxp1, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        relu = tf.nn.relu(fc1) # max(features, 0)
            
        # Apply Dropout
        drop = tf.nn.dropout(relu, dropout)
    
        # Output, class prediction
        out = tf.add(tf.matmul(drop, weights['out']), biases['out'])
        return out
    
    ### modifying conv1 init wigths 
    initialValues_wc1_Rauschen = np.random.normal(0,10,[conv1Kernal_size, 1, conv1Feature_Number])
    initialValues_wc1_values = np.zeros([conv1Kernal_size, 1, conv1Feature_Number], dtype = np.float32)   
    for i in range(0, 32):
        initialValues_wc1_values[:,:,i] = initialValues_wc1_values[:,:,i] + 5 * (16 - i)
    initialValues_wc1 = np.add(initialValues_wc1_values, initialValues_wc1_Rauschen, dtype = np.float32)
    
    # Store layers weight & bias
    weights = {
        # 15x1 conv, 1 input, 32 outputs
        'wc1': tf.Variable(initialValues_wc1),
        # 10x1 conv, 32 inputs, 64 outputs
        'wd1': tf.Variable(tf.random_normal([int(timeSeriesLength/2)*1*conv1Feature_Number, 1024])),
        # fully connected, 1024 inputs, 512 outputs        
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    
    savedWeights = {
        'wc1': np.zeros((conv1Kernal_size, 1, conv1Feature_Number)),
        'wd1': np.zeros((timeSeriesLength*1*conv2Feature_Number, 1024)),
        'out': np.zeros((1024, n_classes))
    }
    
    biases = {
        'bc1': tf.Variable(tf.random_normal([conv1Feature_Number])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    savedBiases = {
        'bc1': np.zeros((conv1Feature_Number)),
        'bd1': np.zeros((1024)),
        'out': np.zeros((n_classes))
    }
    # Construct model
    pred = conv_net(x, weights, biases, keep_prob) # calculates the probabilities of the different types as array
    # scaling to avoid the very big output of matrix multiplications in conv_net
    pred_scaled = pred
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_scaled, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred_scaled, 1), tf.argmax(y, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, initialValues_wc1_values
    
#%%

import tensorflow as tf
import pandas as pd
import numpy as np
import math

tf.reset_default_graph()

# Parameters
learning_rate = 0.01
training_iters = 120000
batch_size = 10000
display_step = 3

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units
conv1Kernal_size = 15
conv1Feature_Number = 128

timeSeriesLength = 360  # time series data input shape: 120 data points
lableNumber = 9
n_classes = 4
## lables:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel-coupon = 1 (12 classes)

#%%

file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
file2Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Data from CIP/Exposure_0.05Quantile_29052017_3.csv'
# Exposure0.05Quantile16052017_coupon_frequency_maturity.csv
# Exposure0.05Quantile16052017_coupon_frequency_maturity_test.csv
# ExposureData05052017maturities6Test
# ExposureData05052017maturities6
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExposureData05052017SmallTest.csv
# ExposureData11052017_coupon_frequency_maturity.csv    

X, lables = loadDataFromCSV(file1Name, False, True, lableNumber) # normalize?, randomize ? 
X_other, lables_other = loadDataFromCSV(file2Name, False, True, lableNumber)

### Stack Data together
# X_data_all_rand = np.row_stack((X_data_all_rand, OtherTestData_rand))
# Y_lable_all_rand = np.row_stack((Y_lable_all_rand, OtherTestLables_rand))

### separate data in test & training
# test_length = 250
test_length = int(len(X) * 0.05) # ...% test data
train_length = len(X) - test_length
X_train = np.float32(X[0: train_length])
X_test = np.float32(X[train_length : len(X)])
lables_train = np.float32(lables[0: train_length])
lables_test = np.float32(lables[train_length : len(X)])
print('Data separated')    

init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, initialValues_wc1_values = Model()

### take batch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size], batch_number * batch_size, batch_number * batch_size + batch_size

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (", conv1Kernal_size*1*conv1Feature_Number + 180*1*conv1Feature_Number*1024 + 1024*n_classes, ")")    
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size <= training_iters:
        
        ### start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0
            r += 1
            # new order of rows of data for next trainings run
            randomizeRows(X_train, lables_train)
            print('##### round ', r+1, "   ", step, " #####")
        batch_x, batch_xa, batch_xb = next_batch(X_train, batch_size, batch_number)
        batch_lables, batch_la, batch_lb = next_batch(lables_train, batch_size, batch_number)
#        print("batch Number: ", batch_number)
#        print("batch ranges: [", batch_xa, ", ", batch_xb, "], [", batch_la, ", ", batch_lb, "]")
        batch_number += 1
        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, 
                                       y: batch_lables,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_lables,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
        ### collect data for TensorBoard
#        with tf.name_scope('cost'):
#            tf.summary.scalar('cost', cost)
#        with tf.name_scope('dropout'):
#            tf.summary.histogram('dropout_keep_probability', keep_prob)

        if(step * batch_size > training_iters - batch_size):
            savedWeights['wc1'] = weights['wc1'].eval()
            savedWeights['wd1'] = weights['wd1'].eval()
            savedWeights['out'] = weights['out'].eval()
            savedBiases['bc1'] = biases['bc1'].eval()
            savedBiases['bd1'] = biases['bd1'].eval()
            savedBiases['out'] = biases['out'].eval()
            print("Weights saved")            
            
            saver = tf.train.Saver()
            save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-kernal120.ckpt")
        
    print("Model saved in file: %s" % save_path)

            
    print("Training Finished!")
    
    # Calculate accuracy for test data
    TestAtEnd = float(sess.run(accuracy, feed_dict={x: X,
                                      y: lables,
                                      keep_prob: 1.}))
    print("Testing Accuracy:", TestAtEnd)

    OtherTest = float(sess.run(accuracy, feed_dict={x: X_other,
                                                    y: lables_other,
                                                    keep_prob: 1.}))
    print("Other Testing Accuracy:", OtherTest)   

# save the run in dict
newEntry = {"test_accuracy" : TestAtEnd,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "conv1Kernal_size" : conv1Kernal_size,
            "conv2Kernal_size" : 0,
            "dropout" : dropout,
            "conv1Feature_Number" : conv1Feature_Number,
            "conv2Feature_Number" : 0,
            "OtherTest_Accuracy" : OtherTest,
            "n_classes" : n_classes,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "comment" : 'incomplete data from CIP; kernal size 120 & 16; only one conv. and one fullyconnected'}  
results.append(newEntry)
  
#%%

tf.reset_default_graph()

# CSV_Name = file1Name
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/O5887085_interpolate_english_normalizedNotional.csv'

X_unlabled = loadDataFromCSVunlabled(CSV_Name, True, False)

init, optimizer, cost, accuracy, x, y, weights, savedWeights2, biases, savedBiases2, keep_prob, pred, initialValues_wc1_values = Model()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)    
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-kernal120.ckpt")
    print("Session restored")    
    logits = sess.run(pred, feed_dict={   x: X_unlabled,
                                            keep_prob: 1.})
    print("Probabilities calculated")  
    logits_scaled = logits # das scaling an dieser Stelle verändert die Wkeiten stark, aber nicht die Reihenfolge
    probabilities = tf.nn.softmax(logits_scaled).eval()


# Stack infos together
probabilities_argmax = np.argmax(probabilities, 1)
toBePrinted = np.column_stack((probabilities_argmax, probabilities))

# print the probabilities into a xlsx
import xlsxwriter
workbook = xlsxwriter.Workbook(CSV_Name[:-4] + '_evaluated.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')

