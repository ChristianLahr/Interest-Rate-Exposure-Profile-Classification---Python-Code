import tensorflow as tf
import pandas as pd
import numpy as np
import math

tf.reset_default_graph()

def randomizeRows(VectData, VectLabel):
    # brings the rows of both vectors in the same new random order
    # both vectors need same length
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return tempData, tempLabel
    
def loadDataFromCSV(path, normalize, randomize, labelNumber): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    ###### select here the labels to be analyzed
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    labels = completeData.iloc[:, completeData.shape[1] - labelNumber : completeData.shape[1] - (labelNumber-1) ].values
    ###### select here the labels to be analyzed

    a = np.zeros((datapoints.shape[0], n_classes)) 
    for i, number in enumerate(labels):
        a[i][ int(number[0]) ] = 1        

    labels_Matrix = a
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    # randomize order
    if randomize == True:
        datapoints_rand_list, labels_rand_list = randomizeRows(datapoints, labels_Matrix)
        datapoints_rand = np.float32(datapoints_rand_list)
        labels_rand = np.float32(labels_rand_list)
    else: 
        datapoints_rand = np.float32(datapoints)
        labels_rand = np.float32(labels_Matrix)
    print('Data loaded')
    return datapoints_rand, labels_rand
    
# Parameters
learning_rate = 0.01
training_iters = 40000
batch_size = 1000
display_step = 1

# Network Parameters
timeSeriesLength = 360  # time series data input shape: 120 data points
labelNumber = 5

conv1Kernal_size = 30
conv1Feature_Number = 32
conv2Kernal_size = 15
conv2Feature_Number = 64
n_classes = 6
    
def Model():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, timeSeriesLength])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # for dropout
    
    def conv1d(x, W, b, strides=1): # Conv1D wrapper
        x = tf.nn.conv1d(x, W, stride=strides, padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x) # ReLu activaton  
        
    def maxpool1d(x, k=2):   # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')    
    
    def conv_net(x, weights, biases, dropout): # create a CNN
        # Reshape input
        x = tf.reshape(x, shape=[-1, timeSeriesLength, 1])
        # Convolution Layer 1
        conv1 = conv1d(x, weights['wc1'], biases['bc1'], strides = 1 )    
        # Max pooling 1 (down-sampling) 
        maxp1 = tf.reshape(conv1, shape=[-1, timeSeriesLength, 1, conv1Feature_Number])
        maxp1 = maxpool1d(maxp1, k=2)
        maxp1 = tf.reshape(maxp1, shape=[-1, int(timeSeriesLength/2), conv1Feature_Number])
        # Convolution Layer 2
        conv2 = conv1d(maxp1, weights['wc2'], biases['bc2'], strides = 1)
        # Max pooling 2 (down-sampling)
        maxp2 = tf.reshape(conv2, shape=[-1, int(timeSeriesLength/2), 1, conv2Feature_Number])
        maxp2 = maxpool1d(maxp2, k=2)        
        # Apply Dropout
        drop = tf.nn.dropout(maxp2, dropout)
        # Reshape maxp2 output to fit fully connected layer input
        fc1 = tf.reshape(drop, [-1, weights['wd1'].get_shape().as_list()[0]])
        # Fully connected layer 1
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        relu = tf.nn.relu(fc1)    
        # Fully connected layer 2
        fc2 = tf.add(tf.matmul(relu, weights['wd2']), biases['bd2'])
        relu = tf.nn.relu(fc2) 
        # Output, class prediction
        out = tf.add(tf.matmul(relu, weights['out']), biases['out'])
        return out
    
    # Define all the variables as dict
    weights = {
        # 15x1 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([conv1Kernal_size, 1, conv1Feature_Number])),
        # 10x1 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([conv2Kernal_size, conv1Feature_Number, conv2Feature_Number])),
        # fully connected,30*1*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([int(timeSeriesLength/4)*1*conv2Feature_Number, 1024])),
        # fully connected, 1024 inputs, 512 outputs        
        'wd2': tf.Variable(tf.random_normal([1024, 512])),
        'out': tf.Variable(tf.random_normal([512, n_classes]))}    
    biases = {
        'bc1': tf.Variable(tf.random_normal([conv1Feature_Number])),
        'bc2': tf.Variable(tf.random_normal([conv2Feature_Number])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([512])),
        'out': tf.Variable(tf.random_normal([n_classes]))}
                                            
    # Model prediction
    pred = conv_net(x, weights, biases, keep_prob)
    # Define loss function (cost) and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Define model evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    # Define the initializer
    init = tf.global_variables_initializer()
    
    return init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, pred

init, optimizer, cost, accuracy, x, y, weights, biases, keep_prob, pred = Model()

### Load labelled data
file1Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
X, labels = loadDataFromCSV(file1Name, False, True, labelNumber) # normalize?, randomize ? 

### separate data in test & training
test_length = int(len(X) * 0.1) # ...% test data
train_length = len(X) - test_length
X_train = np.float32(X[0: train_length])
X_test = np.float32(X[train_length : len(X)])
labels_train = np.float32(labels[0: train_length])
labels_test = np.float32(labels[train_length : len(X)])
print('Data separated')    

### take batch
def next_batch(x, batch_size, batch_number):
    return x[ batch_number * batch_size : batch_number * batch_size + batch_size]

keep_prob = 0.75 # probability to keep a unit!
### Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1; batch_number = 0; r = 0
    stepsToRunThroughTrainingsData = int(train_length / batch_size)
    print("Start training")
    # Keep training until trainings iteration reached
    while step * batch_size < training_iters:
        ### start at the beginning of Trainingsdata after passing it
        if((step - r * stepsToRunThroughTrainingsData) > stepsToRunThroughTrainingsData):
            batch_number = 0; r += 1
            # new order of data for next trainings run
            randomizeRows(X_train, labels_train)
            print('round ', r+1, "   ", step)
        batch_x = next_batch(X_train, batch_size, batch_number)
        batch_y = next_batch(labels_train, batch_size, batch_number)
        batch_number += 1
        ### Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1        
    print("Training Finished!")
    # Calculate accuracy for test data
    TestAtEnd = float(sess.run(accuracy, feed_dict={x: X_test,
                                      y: labels_test,
                                      keep_prob: 1.}))
    print("Testing Accuracy:", TestAtEnd)
