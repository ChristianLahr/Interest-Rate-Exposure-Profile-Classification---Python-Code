#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:01:53 2017

@author: Chris
"""
import tensorflow as tf
import numpy as np
from ownFeatureExtractionModels import Model_Complex_multiDimFeatures, Model_deepNetwork_shorter

def useTrainedModel(networkInput, name_SavesModel, n_classes, numberWeigthsPerNeuronInFirstLayer):
    ###  Parameters (need to be the same as used to train the model!!)
    learning_rate = 0.01
    
    ### Be shure that the old graph is deleted
    tf.reset_default_graph()
            
    ### store features in the rigth way
    networkInput_length = len(networkInput)
    networkInput_length_oneDimFeatures = networkInput_length - 1
    
    networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
    networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
    networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDimFeatures]
    
    ### load the model: "Model_Complex_multiDimFeatures"
    init, optimizer, cost, accuracy, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_1, labels_tf, keep_prob_tf, weights, biases, prediction = Model_Complex_multiDimFeatures(networkInput_length_oneDimFeatures, n_classes, numberWeigthsPerNeuronInFirstLayer, learning_rate)
    
    ### Launch the graph
    with tf.Session() as session:
        session.run(init)    
        saver = tf.train.Saver()
        saver.restore(session, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/" + name_SavesModel)
#        print("Session restored")
        logits = session.run(prediction, feed_dict={inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                           inputData_tf_multiDimFeature_1: networkInput_multiDimFeature_1,
                                           keep_prob_tf: 1.})
#        print("Probabilities calculated")  
        probabilities = tf.nn.softmax(logits).eval()

    return probabilities
        
def useTrainedModel_DeepNetwork(networkInput, name_SavesModel, numberWeigthsPerNeuronInFirstLayer, withLabels):
    withLabels = False

    ###  Parameters (need to be the same as used to train the model!!)
    learning_rate = 0.01
    
    ### Be shure that the old graph is deleted
    tf.reset_default_graph()
            
    networkInput_length = len(networkInput)
    if(withLabels):
        networkInput_length_oneDimFeatures = networkInput_length - 7
        labels_maturity = networkInput[networkInput_length - 3 ]
        labels_frequency = networkInput[networkInput_length - 2 ]
        labels_coupon = networkInput[networkInput_length - 1 ]
    else:
        networkInput_length_oneDimFeatures = networkInput_length - 4        
        
    networkInput_oneDimFeatures = np.array([networkInput[0], networkInput[1], networkInput[2], networkInput[3], networkInput[4], networkInput[5], networkInput[6], networkInput[7], networkInput[8], networkInput[9]])
    networkInput_oneDimFeatures = np.transpose(networkInput_oneDimFeatures)
    networkInput_multiDimFeature_1 = networkInput[networkInput_length_oneDimFeatures]
    networkInput_prob_Maturity = networkInput[networkInput_length_oneDimFeatures + 1]
    networkInput_prob_Frequency = networkInput[networkInput_length_oneDimFeatures + 2]
    networkInput_prob_Coupon = networkInput[networkInput_length_oneDimFeatures + 3]

    n_classes_maturity = 4
    n_classes_frequency = 3
    n_classes_coupon = 6

    ### load the model: "Model_Complex_multiDimFeatures"
    init, optimizer, cost, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all, inputData_tf_oneDimFeatures, inputData_tf_multiDimFeature_jumps, inputData_tf_probabilities_Maturity, inputData_tf_probabilities_Frequency, inputData_tf_probabilities_Coupon, labels_tf_Maturity, labels_tf_Frequency, labels_tf_Coupon, keep_prob_tf, weights, biases, predicted_Maturity_Class, predicted_Frequency_Class, predicted_Coupon_Class, correct_pred_Maturity, correct_pred_Frequency, correct_pred_Coupon, merged = Model_deepNetwork_shorter(  networkInput_length_oneDimFeatures, 
                                              n_classes_maturity, 
                                              n_classes_frequency, 
                                              n_classes_coupon, 
                                              numberWeigthsPerNeuronInFirstLayer, 
                                              learning_rate)

    ### Launch the graph
    with tf.Session() as sess:
        sess.run(init)    
        saver = tf.train.Saver()
        saver.restore(sess, "/Users/Chris/Python/Machine Learning/Masterarbeit1/feature extraction/Saved Models/ownFeatureExtraction_DeepNetwork_shorter_test.ckpt")
        print("Session restored")
        
        if(withLabels):
            classes_Maturity, classes_Frequency, classes_Coupon, correct_preded_Maturity, correct_preded_Frequency, correct_preded_Coupon, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all = sess.run( [ predicted_Maturity_Class, 
                                                                              predicted_Frequency_Class, 
                                                                              predicted_Coupon_Class,
                                                                              correct_pred_Maturity, 
                                                                              correct_pred_Frequency, 
                                                                              correct_pred_Coupon,
                                                                              accuracy_Maturity, 
                                                                              accuracy_Frequency, 
                                                                              accuracy_Coupon, 
                                                                              accuracy_all ], 
                                                                  feed_dict={ inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                                                              inputData_tf_multiDimFeature_jumps: networkInput_multiDimFeature_1,
                                                                              inputData_tf_probabilities_Maturity: networkInput_prob_Maturity,
                                                                              inputData_tf_probabilities_Frequency: networkInput_prob_Frequency,
                                                                              inputData_tf_probabilities_Coupon: networkInput_prob_Coupon,
                                                                              labels_tf_Maturity: labels_maturity,
                                                                              labels_tf_Frequency: labels_frequency,
                                                                              labels_tf_Coupon: labels_coupon,
                                                                              keep_prob_tf: 1. })            
            return classes_Maturity, classes_Frequency, classes_Coupon, correct_preded_Maturity, correct_preded_Frequency, correct_preded_Coupon, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all

        else:            
            classes_Maturity, classes_Frequency, classes_Coupon = sess.run( [ predicted_Maturity_Class, 
                                                                              predicted_Frequency_Class, 
                                                                              predicted_Coupon_Class ], 
                                                                  feed_dict={ inputData_tf_oneDimFeatures: networkInput_oneDimFeatures,
                                                                              inputData_tf_multiDimFeature_jumps: networkInput_multiDimFeature_1,
                                                                              inputData_tf_probabilities_Maturity: networkInput_prob_Maturity,
                                                                              inputData_tf_probabilities_Frequency: networkInput_prob_Frequency,
                                                                              inputData_tf_probabilities_Coupon: networkInput_prob_Coupon,
                                                                              keep_prob_tf: 1. })
        
            print("Classes predicted")  
            return classes_Maturity, classes_Frequency, classes_Coupon
        
        
        