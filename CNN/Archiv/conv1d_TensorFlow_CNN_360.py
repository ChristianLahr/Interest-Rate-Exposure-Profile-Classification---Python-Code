#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:42:39 2017

@author: Chris

A Convolutional Network implementation using TensorFlow library.
"""

# first define results, but only one time. Than all the results are archived there
results=[{}]
#%%

path = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/PosExpExposure_08062017_curve-5.csv'
# Import other test data
chunksize = batch_size
# Import other test data
reader = pd.read_csv(path, header=None, sep=";", chunksize=chunksize, iterator=True)



def get_nextbatch_stream(reader,size):
    try:
        stream = reader.get_chunk(size)
    except:# restart stream
        reader = pd.read_csv(path, header=None, sep=";", chunksize=chunksize, iterator=True)
        stream = reader.get_chunk(size)
    return stream
        
size = 100
stream = get_nextbatch_stream(reader,size)

    
DataBatch = pd.read_csv(path, header=None, sep=";")
###### select here the lables to be analyzed
datapoints_stream = DataBatch.iloc[0:batch_size, : DataBatch.shape[1] - 10 ].values
lables_stream = DataBatch.iloc[:, DataBatch.shape[1] - lableNumber : DataBatch.shape[1] - (lableNumber-1) ].values

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
    # Import data
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
        # Reshape input
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
    
        # Convolution Layer
        conv2 = conv1d(maxp1, weights['wc2'], biases['bc2'], strides = 1)

        # rectivied linear unit (non-linear activation)
        conv2 = tf.nn.relu(conv2)

        # Max Pooling (down-sampling)
        conv2 = tf.reshape(conv2, shape=[-1, int(timeSeriesLength/2), 1, conv2Feature_Number])
    #    conv2 = tf.nn.local_response_normalization(conv2)
        maxp2 = maxpool1d(conv2, k=2)
        maxp2 = tf.reshape(maxp2, shape=[-1, int(timeSeriesLength/4), conv2Feature_Number]) 
        
        # Fully connected layer 1
        # Reshape maxp2 output to fit fully connected layer input
        fc1 = tf.reshape(maxp2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        relu = tf.nn.relu(fc1) # max(x, 0)
    
        # Fully connected layer 2
        fc2 = tf.add(tf.matmul(relu, weights['wd2']), biases['bd2'])
        relu = tf.nn.relu(fc2) # max(x, 0)    
        
        # Fully connected layer 3
        fc3 = tf.add(tf.matmul(relu, weights['wd3']), biases['bd3'])
        relu = tf.nn.relu(fc3) # max(x, 0)    

        # Fully connected layer 4
        fc4 = tf.add(tf.matmul(relu, weights['wd4']), biases['bd4'])
        relu = tf.nn.relu(fc4) # max(x, 0)    

        # Fully connected layer 5
        fc5 = tf.add(tf.matmul(relu, weights['wd5']), biases['bd5'])
        relu = tf.nn.relu(fc5) # max(x, 0)    

        # Apply Dropout
        drop = tf.nn.dropout(relu, dropout)
    
        # Output, class prediction
        out = tf.add(tf.matmul(drop, weights['out']), biases['out'])
        return out
    
    ##### Define all the variables    

    ### check if there are weights in the variable explorer with te right dimension. If true then use them as initial values. 
    ### This shoud be good, because ... see: https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks
    # if ('weights' in globals())  :
        
"""    
    ### modifying conv1 init wigths 
    initialValues_wc1_Rauschen = np.random.normal(0,10,[conv1Kernal_size, 1, conv1Feature_Number])
    initialValues_wc1_values = np.zeros([conv1Kernal_size, 1, conv1Feature_Number], dtype = np.float32)   
    for i in range(0, conv1Feature_Number):
        initialValues_wc1_values[:,:,i] = initialValues_wc1_values[:,:,i] + 5 * (conv1Feature_Number - i)
    initialValues_wc1 = np.add(initialValues_wc1_values, initialValues_wc1_Rauschen, dtype = np.float32)

    ### modifying conv2 init wigths 
    initialValues_wc2_Rauschen = np.random.normal(0,10,[conv2Kernal_size, conv1Feature_Number, conv2Feature_Number])
    initialValues_wc2_values = np.zeros([conv2Kernal_size, conv1Feature_Number, conv2Feature_Number], dtype = np.float32)   
    for i in range(0, int(conv2Feature_Number/2)):
        initialValues_wc2_values[int(conv2Kernal_size/conv2Feature_Number*i):,:,i] = 0
    for i in range(int(conv2Feature_Number/2), conv2Feature_Number):
        initialValues_wc2_values[:int(conv2Kernal_size/conv2Feature_Number*i),:,i] = 0
    initialValues_wc2 = np.add(initialValues_wc2_values, initialValues_wc2_Rauschen, dtype = np.float32)
"""    
    fullyConnectedStartSize = 512
    weights = {
        # 15x1 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv1Kernal_size, 1, conv1Feature_Number])),
        # 10x1 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([conv2Kernal_size, conv1Feature_Number, conv2Feature_Number])),
        # fully connected,30*1*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([int(timeSeriesLength/4)*1*conv2Feature_Number, fullyConnectedStartSize])),
        # fully connected, 1024 inputs, 512 outputs        
        'wd2': tf.Variable(tf.random_normal([fullyConnectedStartSize, int(fullyConnectedStartSize/2)])),
        # fully connected, 512 inputs, 256 outputs        
        'wd3': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/2), int(fullyConnectedStartSize/4)])),
        # fully connected, 256 inputs, 128 outputs        
        'wd4': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/4), int(fullyConnectedStartSize/8)])),
        # fully connected, 256 inputs, 128 outputs        
        'wd5': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/8), int(fullyConnectedStartSize/16)])),
        # 64 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/16), n_classes]))
    }
    
    savedWeights = {
        'wc1': np.zeros((conv1Kernal_size, 1, conv1Feature_Number)),
        'wc2': np.zeros((conv2Kernal_size, conv1Feature_Number, conv2Feature_Number)),
        'wd1': np.zeros((timeSeriesLength*1*conv2Feature_Number, fullyConnectedStartSize)),
        'wd2': np.zeros((fullyConnectedStartSize, int(fullyConnectedStartSize/2))),
        'out': np.zeros((int(fullyConnectedStartSize/2), n_classes))
    }
    
    biases = {
        'bc1': tf.Variable(tf.random_normal([conv1Feature_Number])),
        'bc2': tf.Variable(tf.random_normal([conv2Feature_Number])),
        'bd1': tf.Variable(tf.random_normal([fullyConnectedStartSize])),
        'bd2': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/2)])),
        'bd3': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/4)])),
        'bd4': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/8)])),
        'bd5': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/16)])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    savedBiases = {
        'bc1': np.zeros((conv1Feature_Number)),
        'bc2': np.zeros((conv2Feature_Number)),
        'bd1': np.zeros((fullyConnectedStartSize)),
        'bd2': np.zeros((int(fullyConnectedStartSize/2))),
        'out': np.zeros((n_classes))
    }
    # Construct model
    pred = conv_net(x, weights, biases, keep_prob) # calculates the probabilities of the different types as array

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, initialValues_wc1_values, correct_pred
    
#%%

import tensorflow as tf
import pandas as pd
import numpy as np
import math

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
training_iters = 4000
batch_size = 1000
display_step = 1

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units
conv1Kernal_size = 15
conv1Feature_Number = 16
conv2Kernal_size = 30
conv2Feature_Number = 32

timeSeriesLength = 360  # time series data input shape: 120 data points
lableNumber = 5
n_classes = 6
## lables:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

#%%

file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
file2Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_29052017_11_quarterly.csv'
# Exposure0.05Quantile16052017_coupon_frequency_maturity.csv
# Exposure0.05Quantile16052017_coupon_frequency_maturity_test.csv
# ExposureData05052017maturities6Test
# ExposureData05052017maturities6
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExposureData05052017SmallTest.csv
# ExposureData11052017_coupon_frequency_maturity.csv    

X, lables = loadDataFromCSV(file1Name, False, True, lableNumber) # normalize?, randomize ? 
X_other, lables_other = loadDataFromCSV(file2Name, False, False, lableNumber)

### Stack Data together
# X_data_all_rand = np.row_stack((X_data_all_rand, OtherTestData_rand))
# Y_lable_all_rand = np.row_stack((Y_lable_all_rand, OtherTestLables_rand))

### separate data in test & training
# test_length = 250
test_length = int(len(X) * 0.1) # ...% test data
train_length = len(X) - test_length
X_train = np.float32(X[0: train_length])
X_test = np.float32(X[train_length : len(X)])
lables_train = np.float32(lables[0: train_length])
lables_test = np.float32(lables[train_length : len(X)])
print('Data separated')    

init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, initialValues_wc1_values, correct_pred = Model()

### take batch
def next_batch(x, batch_size, batch_number):
    try:
        stream = reader.get_chunk(size)
    except:# restart stream
        reader = pd.read_csv(path, header=None, sep=";", chunksize=chunksize, iterator=True)
        stream = reader.get_chunk(size)
    return stream

    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (", conv1Kernal_size*1*conv1Feature_Number + conv2Kernal_size*conv1Feature_Number*conv2Feature_Number + 30*1*conv2Feature_Number*1024 + 1024*512 + 512*n_classes, ")")    
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        ### start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0
            r += 1
            # new order of rows of data for next trainings run
#            randomizeRows(X_train, lables_train)
            print('round ', r+1, "   ", step)
        batch_x = next_batch(X_train, batch_size, batch_number)
        batch_y = next_batch(lables_train, batch_size, batch_number)
        batch_number += 1
        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

        if(step * batch_size >= training_iters - batch_size):
            saver = tf.train.Saver()
            save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-test.ckpt")
    print("Model saved in file: %s" % save_path)

#            savedWeights['wc1'] = weights['wc1'].eval()
#            savedWeights['wc2'] = weights['wc2'].eval()
#            savedWeights['wd1'] = weights['wd1'].eval()
#            savedWeights['wd2'] = weights['wd2'].eval()
#            savedWeights['out'] = weights['out'].eval()
#            savedBiases['bc1'] = biases['bc1'].eval()
#            savedBiases['bc2'] = biases['bc2'].eval()
#            savedBiases['bd1'] = biases['bd1'].eval()
#            savedBiases['bd2'] = biases['bd2'].eval()
#            savedBiases['out'] = biases['out'].eval()
#            print("Weights saved")            
        
    print("Training Finished!")
    
    # Calculate accuracy for test data
    TestAtEnd, correct_pred_testData = sess.run([accuracy, correct_pred], feed_dict={x: X_test, y: lables_test, keep_prob: 1.})
    print("Testing Accuracy:", TestAtEnd)

    OverallDataTest, correct_pred_overallData = sess.run([accuracy, correct_pred], feed_dict={x: X, y: lables, keep_prob: 1.})
    print("Accuracy of overall Data:", OverallDataTest)   

# save the run in dict
newEntry = {"test_accuracy" : TestAtEnd,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "conv1Kernal_size" : conv1Kernal_size,
            "conv2Kernal_size" : conv2Kernal_size,
            "dropout" : dropout,
            "conv1Feature_Number" : conv1Feature_Number,
            "conv2Feature_Number" : conv2Feature_Number,
            "OtherTest_Accuracy" : OtherTest,
            "n_classes" : n_classes,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "comment" : 'incomplete data from CIP'}  
results.append(newEntry)

#%%
print("Number of representations in classes (test data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, lables_test.shape[0]) if (correct_pred_testData[k] == True and lables_test[k,i] == 1))
    number = sum(np.transpose(lables_test)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")    

print("Number of representations in classes (all data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, lables.shape[0]) if (correct_pred_overallData[k] == True and lables[k,i] == 1))
    number = sum(np.transpose(lables)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")    
    
#%%

tf.reset_default_graph()

# CSV_Name = file1Name
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/Data from CIP/O5887085_interpolated.csv'

X_unlabled = loadDataFromCSVunlabled(CSV_Name, True, False)

init, optimizer, cost, accuracy, x, y, weights, savedWeights2, biases, savedBiases2, keep_prob, pred, initialValues_wc1_values = Model()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)    
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-CIP-Data-Frequency.ckpt")
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
workbook = xlsxwriter.Workbook(CSV_Name[:-4] + '_evaluated.xlsx')
worksheet = workbook.add_worksheet('Results Analysis')
row = 0
for col, data in enumerate(np.transpose(toBePrinted)):
    worksheet.write_column(row, col, data)

workbook.close()
print('Analysis written to xlsx')

