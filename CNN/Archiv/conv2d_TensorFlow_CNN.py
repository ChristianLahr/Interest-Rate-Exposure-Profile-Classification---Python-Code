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
    
def loadDataFromCSV(path, normalize, maturityStyle, randomize): # normaliz = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:, : completeData.shape[1]-2 ].values
    lables = completeData.iloc[:, completeData.shape[1]-2 : completeData.shape[1] - 1 ].values

    # maturities labels have be of the form 1, 2, ...
    a = np.zeros((datapoints.shape[0], n_classes)) 
    for i, number in enumerate(lables):
        if(maturityStyle == 'only 6'):
            # if maturisies 1, 3, 5, 7, ...
            a[i][ int((number[0] - 1)/2) ] = 1        
        else: 
            # normal maturities 1, 2, 3, 4, 5, ...
            a[i][ int(number[0] - 1) ] = 1

    lables_Matrix = a
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            maximum = np.ndarray.max(datapoints[i])
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, lables_rand_list = randomizeRows(datapoints, lables_Matrix)
        datapoints_rand = np.float32(datapoints_rand_list)
        lables_rand = np.float32(lables_rand_list)
    print('Data loaded')
    return datapoints_rand, lables_rand
#%%

import tensorflow as tf
import pandas as pd
import numpy as np

# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 25
display_step = 4

maturityStyle = 'normal'

# Network Parameters
n_input = 120 # time searies data input shape: 120 data points
dropout = 0.75 # Dropout, probability to keep units
conv1Kernal_size = 7
conv1Feature_Number = 32
conv2Kernal_size = 20
conv2Feature_Number = 64

# 1-15 maturity
if(maturityStyle == 'only 6'):
    n_classes = 6
else: 
    n_classes = 15


file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/ExpPosExpo_manyParameters-JavaExport_clean.csv'
file2Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/ExposureData05052017SmallTest.csv'
# ExposureData05052017maturities6Test
# ExposureData05052017maturities6
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExpPosExpo_manyParameters-JavaExport_clean.csv
# ExposureData05052017SmallTest.csv
    
X_data_all_rand, Y_lable_all_rand = loadDataFromCSV(file1Name, True, maturityStyle, True)
OtherTestData_rand, OtherTestLables_rand = loadDataFromCSV(file2Name, True, maturityStyle, True)

# Stack Data together
# X_data_all_rand = np.row_stack((X_data_all_rand, OtherTestData_rand))
# Y_lable_all_rand = np.row_stack((Y_lable_all_rand, OtherTestLables_rand))

# separate data in test & training
test_length = 250
# test_length = int(len(X_data_all_rand) * 0,1)
train_length = len(X_data_all_rand) - test_length
X_data_train = np.float32(X_data_all_rand[0: train_length])
X_data_test = np.float32(X_data_all_rand[train_length : len(X_data_all_rand)])
X_lables_train = np.float32(Y_lable_all_rand[0: train_length])
X_lables_test = np.float32(Y_lable_all_rand[train_length : len(X_data_all_rand)])
    
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # see padding explanation here http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    x = tf.nn.conv2d(x, W, strides=[1,strides, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):   
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 120, 1], strides)

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling)
#   conv1 = maxpool2d(conv1, k=1) # k sollte eigentlich 2 sein aber hier keine 2d Daten

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
#    conv2 = maxpool2d(conv2, k=1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1) # max(features, 0)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x1 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([conv1Kernal_size, 1, conv1Feature_Number])),
    # 5x1 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([conv2Kernal_size, conv1Feature_Number, conv2Feature_Number])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([120*1*conv2Feature_Number, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
savedWeights = {
    'wc1': np.zeros((conv1Kernal_size, 1, conv1Feature_Number)),
    'wc2': np.zeros((conv2Kernal_size, conv1Feature_Number, conv2Feature_Number)),
    'wd1': np.zeros((120*1*conv2Feature_Number, 1024)),
    'out': np.zeros((1024, n_classes))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([conv1Feature_Number])),
    'bc2': tf.Variable(tf.random_normal([conv2Feature_Number])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

savedBiases = {
    'bc1': np.zeros((conv1Feature_Number)),
    'bc2': np.zeros((conv2Feature_Number)),
    'bd1': np.zeros((1024)),
    'out': np.zeros((n_classes))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob) # calculates the probabilities of the different types as array
# scaling to avoid the very big output of matrix multiplications in conv_net
pred_scaled = pred / 10000
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_scaled, labels=y))

listOfVariablesToBeChanged = [weights['wc1'], weights['wc2'], weights['wd1'], weights['out'], biases['bc1'], biases['bc2'], biases['bd1'], biases['out']]
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list = listOfVariablesToBeChanged)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average

# Initializing the variables
init = tf.global_variables_initializer()

# take patch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized")    
    step = 1
    batch_number = 0
    r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        # start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0
            r += 1
            # new order of rows of data for next trainings run
            randomizeRows(X_data_train, X_lables_train)
            print('round ', r+1, "   ", step)
        batch_x = next_batch(X_data_train, batch_size, batch_number)
        batch_y = next_batch(X_lables_train, batch_size, batch_number)
        batch_number += batch_number
        # Run optimization op (backprop)
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
        
        # collect data for TensorBoard
#        with tf.name_scope('cost'):
#            tf.summary.scalar('cost', cost)
#        with tf.name_scope('dropout'):
#            tf.summary.histogram('dropout_keep_probability', keep_prob)

        if(step * batch_size >= training_iters - batch_size):
            savedWeights['wc1'] = weights['wc1'].eval()
            savedWeights['wc2'] = weights['wc2'].eval()
            savedWeights['wd1'] = weights['wd1'].eval()
            savedWeights['out'] = weights['out'].eval()
            savedBiases['bc1'] = biases['bc1'].eval()
            savedBiases['bc2'] = biases['bc2'].eval()
            savedBiases['bd1'] = biases['bd1'].eval()
            savedBiases['out'] = biases['out'].eval()
            print("Weights saved")            
            
    print("Training Finished!")

    # Calculate accuracy for test data
    TestAtEnd = float(sess.run(accuracy, feed_dict={x: X_data_test,
                                      y: X_lables_test,
                                      keep_prob: 1.}))
    print("Testing Accuracy:", TestAtEnd)

    OtherTest = float(sess.run(accuracy, feed_dict={x: OtherTestData_rand,
                                                    y: OtherTestLables_rand,
                                                    keep_prob: 1.}))
    print("Other Testing Accuracy:", OtherTest)
    
    # merge and store the summary
#    merged = tf.summary.merge_all()
#    train_writer = tf.summary.FileWriter('summaryData', sess.graph)
    
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
            "comment" : ''}  
results.append(newEntry)


  
#%%
# print results
for k, entry in enumerate(results):
    if(k>0):
        print("Accuracy ", entry["test_accuracy"])

#%%

# Parameters
learning_rate = 0.001
training_iters = 200
batch_size = 50
display_step = 4
n_input = 120 # time searies data input (shape: 120 data points)
n_classes = 15 # 1-15 maturity

# important parameters
learning_rate = 0.001
training_iters = 2000
dropout = 0.75
conv1Kernal_size = 15
conv2Kernal_size = 15
conv1Feature_Number = 64
conv2Feature_Number = 128

#%%

def predictData(CSV_Name, maturityStyle, randomize):

    fileName_External = CSV_Name
    externalTestDataPoints, externalTestLables = loadDataFromCSV(fileName_External, True, maturityStyle, randomize)
    
    # maturityStyle = 'only 6' # 'normal': 15 classes; other: 'only 6'
    
    if(maturityStyle == 'only 6'):
        n_classes = 6
    else: 
        n_classes = 15
    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    
    # restore layers weight & bias
    weights_External = {
        'wc1': tf.Variable(savedWeights['wc1']),
        'wc2': tf.Variable(savedWeights['wc2']),
        'wd1': tf.Variable(savedWeights['wd1']),
        'out': tf.Variable(savedWeights['out'])}
    biases_External = {
        'bc1': tf.Variable(savedBiases['bc1']),
        'bc2': tf.Variable(savedBiases['bc2']),
        'bd1': tf.Variable(savedBiases['bd1']),
        'out': tf.Variable(savedBiases['out'])}
    print('weights loaded')
    
    # Construct model
    pred_External = conv_net(x, weights_External, biases_External, keep_prob) # calculates the probabilities of the different types as array
    # scaling
    pred_External_scaled = pred_External / 10000

    # Evaluate model
    correct_pred_External = tf.equal(tf.argmax(pred_External_scaled, 1), tf.argmax(y, 1)) # argmax gibt die Stelle an der die 1 steht; danach wird verglichen ob sie bei den beiden an der gleichen Stelle ist. Wenn ja --> richtig geschätzt also true sonst false. Also entsthet ein Vector[Booleans]    
    accuracy_External = tf.reduce_mean(tf.cast(correct_pred_External, tf.float32)) # cast tensor to tf.float32 type and calculates the overall mean -> how many right predictions on average
    
    # Initializing the variables
    init_External = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init_External)
        print('Network initialized')
        externalTest, pred_External_Vector = sess.run([accuracy_External, pred_External_scaled], feed_dict={x: externalTestDataPoints,
                                                        y: externalTestLables,
                                                        keep_prob: 1.})
        print("External Accuracy:", float(externalTest))
        
    return pred_External_Vector
            
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/ExposureData05052017SmallTest.csv'
maturityStyle = 'normal'
predictedProbabilities_scaled_logits = predictData(CSV_Name, maturityStyle, True)
print(predictedProbabilities_scaled_logits)
#%%
with tf.Session() as sess:
    predictedProbabilities_softmax = tf.nn.softmax(predictedProbabilities_scaled_logits).eval()
#    print(predictedProbabilities_softmax)

#%%

with tf.Session() as sess:
    predictedProbabilities_fixed = tf.nn.softmax(predictedProbabilities_scaled_logits, -1).eval()

    print('done')
    