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

"""
path = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/PosExpExposure_08062017_curve-5.csv'

reader = pd.read_csv(path, header=None, sep=";", chunksize=chunksize, iterator=True)
def get_nextbatch_stream(reader,size):
    stream = reader.get_chunk(size)
    datapoints  = stream.iloc[:, : stream.shape[1] - 10 ].values
    lables      = stream.iloc[:, stream.shape[1] - lableNumber : completeData.shape[1] - (lableNumber-1) ].values
    return datapoints, lables
        
size = 100
try:
    stream_p, stream_l = get_nextbatch_stream(reader,size)
except StopIteration:
    reader = pd.read_csv(path, header=None, sep=";", chunksize=chunksize, iterator=True)
    stream_p, stream_l = get_nextbatch_stream(reader,size)
    
DataBatch = pd.read_csv(path, header=None, sep=";")
###### select here the lables to be analyzed
datapoints_stream = DataBatch.iloc[0:batch_size, : DataBatch.shape[1] - 10 ].values
lables_stream = DataBatch.iloc[:, DataBatch.shape[1] - lableNumber : DataBatch.shape[1] - (lableNumber-1) ].values
"""

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
    fullyConnectedStartSize = 512
    weightConstnatSmaller = 0.1
    weights = {
        # 15x1 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv1Kernal_size, 1, conv1Feature_Number]) * weightConstnatSmaller),
        # 10x1 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([conv2Kernal_size, conv1Feature_Number, conv2Feature_Number]) * weightConstnatSmaller),
        # fully connected,30*1*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([int(timeSeriesLength/4)*1*conv2Feature_Number, fullyConnectedStartSize]) * weightConstnatSmaller),
        # fully connected, 1024 inputs, 512 outputs        
        'wd2': tf.Variable(tf.random_normal([fullyConnectedStartSize, int(fullyConnectedStartSize/2)]) * weightConstnatSmaller),
        # fully connected, 512 inputs, 256 outputs        
        'wd3': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/2), int(fullyConnectedStartSize/4)]) * weightConstnatSmaller),
        # fully connected, 256 inputs, 128 outputs        
        'wd4': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/4), int(fullyConnectedStartSize/8)]) * weightConstnatSmaller),
        # fully connected, 256 inputs, 128 outputs        
        'wd5': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/8), int(fullyConnectedStartSize/16)]) * weightConstnatSmaller),
        # 64 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([int(fullyConnectedStartSize/16), n_classes]) * weightConstnatSmaller)
    }
    
    savedWeights = {
        'wc1': np.zeros((conv1Kernal_size, 1, conv1Feature_Number)),
        'wc2': np.zeros((conv2Kernal_size, conv1Feature_Number, conv2Feature_Number)),
        'wd1': np.zeros((timeSeriesLength*1*conv2Feature_Number, fullyConnectedStartSize)),
        'wd2': np.zeros((fullyConnectedStartSize, int(fullyConnectedStartSize/2))),
        'out': np.zeros((int(fullyConnectedStartSize/2), n_classes))
    }
    
    biases = {
        'bc1': tf.Variable(tf.zeros((conv1Feature_Number))),
        'bc2': tf.Variable(tf.zeros((conv2Feature_Number))),
        'bd1': tf.Variable(tf.zeros((fullyConnectedStartSize))),
        'bd2': tf.Variable(tf.zeros((int(fullyConnectedStartSize/2)))),
        'bd3': tf.Variable(tf.zeros((int(fullyConnectedStartSize/4)))),
        'bd4': tf.Variable(tf.zeros((int(fullyConnectedStartSize/8)))),
        'bd5': tf.Variable(tf.zeros((int(fullyConnectedStartSize/16)))),
        'out': tf.Variable(tf.zeros((n_classes)))
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

    return init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred, tf.nn.softmax(pred), correct_pred
    
#%%

import tensorflow as tf
import pandas as pd
import numpy as np

tf.reset_default_graph()

# Parameters
learning_rate = 0.001
training_iters = 150000
batch_size = 5000
display_step = 1

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units
conv1Kernal_size = 15
conv1Feature_Number = 16
conv2Kernal_size = 30
conv2Feature_Number = 32

timeSeriesLength = 360  # time series data input shape: 120 data points
lableNumber = 9
n_classes = 4
## lables:
# maturity = 9      (4 classes)
# frequency = 7     (3 classes)
# coupon = 5        (6 classes)
# cuveLevels = 3    (6 classes)
# Difference CurveLevel; coupon = 1 (12 classes)

#%%

file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
file2Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_29052017_11_quarterly.csv'

init, optimizer, cost, accuracy, x, y, weights, savedWeights, biases, savedBiases, keep_prob, pred2, pred, correct_pred = Model()

### take batch
def next_batch(reader, size, n_classes, normalize):
    stream = reader.get_chunk(size)
    # separate labels
    datapoints  = stream.iloc[:, : stream.shape[1] - 10 ].values
    lables      = stream.iloc[:, stream.shape[1] - lableNumber : stream.shape[1] - (lableNumber-1) ].values
    # create one hot vector for labels
    a = np.zeros((datapoints.shape[0], n_classes)) 
    for i, number in enumerate(lables):
        a[i][ int(number[0]) ] = 1        

    lables = a
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    return datapoints, lables
    
reader = pd.read_csv(file1Name, header=None, sep=";", chunksize=batch_size, iterator=True)

### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print("Network variables initialized (>", conv1Kernal_size*1*conv1Feature_Number + conv2Kernal_size*conv1Feature_Number*conv2Feature_Number + 512 * 512 /2, ")")    
    step = 1
    print("Start training")
    # Keep training until reach max iterations
    while step * batch_size <= training_iters:
        
        try:
            batch_x, batch_y = next_batch(reader, batch_size, n_classes, False)
        except StopIteration:
            reader = pd.read_csv(file1Name, header=None, sep=";", chunksize=batch_size, iterator=True)
            batch_x, batch_y = next_batch(reader, batch_size, n_classes, False)

        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, predictionss, predictionss2 = sess.run([cost, accuracy, pred, pred2], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    saver = tf.train.Saver()
    save_path = saver.save(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-360-stream-Coupon-test.ckpt")
    print("Model saved in file: %s" % save_path)
        
    print("Training Finished!")
    
    # big test batch

    reader = pd.read_csv(file1Name, header=None, sep=";", chunksize=50000, iterator=True)
    batch_x, batch_y = next_batch(reader, 500, n_classes, False)
    
    # Calculate accuracy for test data
    TestAtEnd, correct_pred_testData = sess.run([accuracy, correct_pred], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print("Testing Accuracy:", TestAtEnd)

    OverallDataTest = TestAtEnd
    correct_pred_overallData = correct_pred_testData

print("Number of representations in classes (all data)")
for i in range(0, n_classes):
    correct = sum(1 for k in range(0, batch_y.shape[0]) if (correct_pred_overallData[k] == True and batch_y[k,i] == 1))
    number = sum(np.transpose(batch_y)[i])
    print("class ", i, ": ", number, "\t correct: ", correct, "\t", float(np.round(correct/number*100, 3)), "%")    
    
#%%
print(predictionss[0])
print(predictionss2[0])

#%%
tf.reset_default_graph()

# CSV_Name = file1Name
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/All Data/O5887085_interpolate_english.csv'

X_unlabled = loadDataFromCSVunlabled(CSV_Name, True, False)

init, optimizer, cost, accuracy, x, y, weights, savedWeights2, biases, savedBiases2, keep_prob, pred, initialValues_wc1_values = Model()

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/CNN/trainedModels/CNN-test.ckpt")
    print("Session restored")    
    logits = sess.run(pred, feed_dict={x: X_unlabled, keep_prob: 1.})
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

