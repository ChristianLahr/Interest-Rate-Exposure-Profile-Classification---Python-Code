#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:54:53 2017

@author: Chris
"""

###### Parameters

learning_rate = 0.01
training_iters = 1000000
batch_size = 1000
display_step = 10
n_hidden = 128 # hidden layer num of features for RNN
n_steps = 360

numberWeigthsPerNeuronInFirstLayer = 100
featuresNeuBerechnenUndDataNeuLaden = False

lableNumber = 5
n_classes = 6
dropout = 0.9
## lables:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)


import tensorflow as tf
import pandas as pd
import numpy as np
from ownFeatureExtractionModels import Model_easy_RNN


def randomizeRows(VectData, VectLabel):
    # brings the rows of both vectors in the same new random order
    # both vectors need same length
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return tempData, tempLabel

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
            # also normalizes negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            if(maximum == 0): maximum = 1 #if it is a zero line we need no normalization (avoid deviding througth zero)
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

### Load Exposure Profiles
if(featuresNeuBerechnenUndDataNeuLaden):
    file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Test Data/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
    data, lables = loadDataFromCSV(file1Name, False, False, lableNumber) # normalize?, randomize ? 
    data = data[:2000, :]
    lables = lables[:2000,:]

    ### clear the data from zero lines
    zeroLine = np.zeros(360)
    badIndexes = []
    for index, item in enumerate(data):
    #    if(index==0): print(item)
        if(np.array_equal(item, zeroLine)): badIndexes.append(index)
    for i in reversed(badIndexes):
        data = np.delete(data, i, 0)
        lables = np.delete(lables, i, 0)

    timeSeriesLength = data.shape[1]
    numberOfProfiles = data.shape[0]

    ## Feature 1 & 2: Maximum & Position of the maximum
    print("calclate features 1 & 2")
    maximum = np.zeros(numberOfProfiles, dtype = np.float32)
    maximumPosition = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses
        maximum[i] = max(data[i])
        negmaximum = max(-data[i])
        if(negmaximum > maximum[i]): maximum[i] = negmaximum
        for position, value in enumerate(data[i]):   #Loop over time steps
            if(value == maximum[i]): maximumPosition[i] = position
    
    ## Feature 3: First Point
    print("calclate features 3")
    entryPoint = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        entryPoint[i] = data[i, 0]
    
    ## Feature 4: Delta(maximum, first point)
    print("calclate features 4")
    deltaMaxFist = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        deltaMaxFist[i] = maximum[i] - entryPoint[i]
    
    ## Feature 5: delta to the next point (first derivative)
    print("calclate features 5")
    delta = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        for j in range(0, timeSeriesLength - 1):
            delta[i,j] = data[i,j] - data[i,j+ 1]
    
    ## Feature 6: max(delta, 0) = down jumps
    print("calclate features 6")
    jumps = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        for j in range(0, timeSeriesLength - 1):
            jumps[i,j] = max(delta[i,j], 0)
    
    ## Feature 7: time passed after last jump
    print("calclate features 7")
    timeAfterJump = np.zeros((numberOfProfiles, 359), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        time = 0
        for j in range(0, timeSeriesLength - 1):
            if(jumps[i,j] == 0): time = time +1
            if(jumps[i,j] > 0): time = 0
            timeAfterJump[i,j] = time
    
    ## Feature 8: number of zeros at the end (length of the zero line at the end)
    print("calclate features 8")
    numberOfZeros = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfZeros[i] = timeAfterJump[i, 358] #there are only 359 jumps --> index 358
        if(data[i,359] != 0): numberOfZeros[i] = 0
    
    ## Feature 9: maximal distance to the next drop3
    print("calclate features 9")
    maximalDistanceToTheNextDrop = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses   
        maximalDistanceToTheNextDrop[i] = max(timeAfterJump[i,int(maximumPosition[i]):int(360-numberOfZeros[i])])
    
        
    ## Feature 10: mean of the tensor without last zeros
    print("calclate features 10")
    mean = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        mean[i] = np.mean(data[i])
    
    ## Feature 11: variance of the tensor without last zeros
    print("calclate features 11")
    variance = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        variance[i] = np.var(data[i])  

    ## Feature 12 & 13: number of jumps and average jump higth
    print("calclate features 12 & 13")
    numberOfJumps = np.zeros(numberOfProfiles, dtype = np.float32)
    averageJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        number_temp = 0
        sum_temp = 0
        for j in range(0, timeSeriesLength - 1):
            if(jumps[i,j] > 0): 
                number_temp += 1
                sum_temp += jumps[i,j]
        numberOfJumps[i] = number_temp
        if(number_temp == 0):   averageJump[i] = 0
        if(number_temp > 0):    averageJump[i] = sum_temp / number_temp
        
    ## Feature 14: number of jumps relative to net (real) time series length
    print("calclate features 14")
    numberOfJumpsRelativeToLength = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfJumpsRelativeToLength[i] = numberOfJumps[i] / (360-numberOfZeros[i])
    
    ## Feature 15: highest jump
    print("calclate features 15")
    maximalJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        maximalJump[i] = max(jumps[i])
        

    print("Features calculated")
    
networkInput = np.array([maximum, maximumPosition, entryPoint, numberOfZeros, mean, variance, maximalDistanceToTheNextDrop, deltaMaxFist])
networkInput = np.transpose(networkInput)
networkInput_length = networkInput.shape[1]

print("use", networkInput_length, "features as network input")

init, optimizer, cost, accuracy, rawData_tf, inputData_tf, lables_tf, keep_prob_tf, weights, biases, prediction = Model_easy_RNN(networkInput_length, n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate, n_steps, n_hidden)

### take batch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

train_length = len(data)

### randomize data before the traing start
#networkInput, lables = randomizeRows(networkInput, lables)

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (", numberWeigthsPerNeuronInFirstLayer * 6 + numberWeigthsPerNeuronInFirstLayer * numberWeigthsPerNeuronInFirstLayer/2 + numberWeigthsPerNeuronInFirstLayer/2 * numberWeigthsPerNeuronInFirstLayer/4 + numberWeigthsPerNeuronInFirstLayer/4 * numberWeigthsPerNeuronInFirstLayer/10 + numberWeigthsPerNeuronInFirstLayer/10 * n_classes, ")")    
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
            # new order of rows of data for next training run
#            networkInput, lables = randomizeRows(networkInput, lables)
            print('round ', r+1, "   ", step)
        batch_x = next_batch(networkInput, batch_size, batch_number)
        batch_rawdata = next_batch(data, batch_size, batch_number)
        batch_y = next_batch(lables, batch_size, batch_number)
        batch_number += 1
        ### Run optimization op (backprop)
        
        print(batch_rawdata.shape())
        sess.run(optimizer, feed_dict={rawData_tf: batch_rawdata,
                                       inputData_tf: batch_x, 
                                       lables_tf: batch_y,
                                       keep_prob_tf: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={  rawData_tf: batch_rawdata,
                                                                inputData_tf: batch_x, 
                                                                lables_tf: batch_y,
                                                                keep_prob_tf: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        
        """
        ### save the model after last iteration
        if(step * batch_size >= training_iters - batch_size):            
            saver = tf.train.Saver()    
            save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_firstTry.ckpt")
            print("Model saved in file: %s" % save_path)
        """
            
    print("Training Finished!")
    
    # Calculate accuracy for test data
    overallTest = float(sess.run(accuracy, feed_dict={rawData_tf: data,
                                                      inputData_tf: networkInput,
                                                      lables_tf: lables,
                                                      keep_prob_tf: 1.}))
    print("Testing Accuracy:", overallTest)
    
    anotherTest = overallTest
#    anotherTest = float(sess.run(accuracy, feed_dict={inputData_tf: networkInput,
#                                                    lables_tf: lables,
#                                                    keep_prob_tf: 1.}))
    print("Other Testing Accuracy:", anotherTest)   

# save the run in dict
newEntry = {"test_accuracy" : overallTest,
            "learning_rate" : learning_rate,
            "training_iters" : training_iters,
            "dropout" : dropout,
            "OtherTest_Accuracy" : anotherTest,
            "n_classes" : n_classes,
            "train_length" : train_length,
            "batch_size" : batch_size,
            "numberWeigthsPerNeuronInFirstLayer": numberWeigthsPerNeuronInFirstLayer,
            "lableNumber": lableNumber,
            "Model": 'Model_together',
            "comment" : '6 Input jeweils mit ... Weigths und dann 3 fully connected'}  
results.append(newEntry)


# Ideen:
    #erst sollen alle features alleine eine klasse wählen, dann diese info weiterverwenden zusätzlich zu den infos die man davor hatte
    #alle features wählen klasse zusammen 6 inputs --> weights --> class outputs
