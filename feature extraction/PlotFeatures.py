#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:32:05 2017

@author: Chris
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path_Features = '/Users/Chris/Python/Machine Learning/Masterarbeit_Git/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'
path_save = '/Users/Chris/Desktop/FeaturePlots/'

def next_batch_ofFeatures(reader, size):
    n_classes_maturity  = 4
    n_classes_frequency = 3
    n_classes_coupon    = 6
    
    labelNumber_maturity = 9
    labelNumber_frequency = 7
    labelNumber_coupon = 5

    stream = reader.get_chunk(size)
    # separate labels
    n_all = n_classes_maturity + n_classes_frequency + n_classes_coupon
    n_feature = stream.shape[1] - 10 -  n_all
    features            = stream.iloc[:, : n_feature].values
    probs               = stream.iloc[:, n_feature : n_feature + n_all ].values
    labels_maturity     = stream.iloc[:, stream.shape[1] - labelNumber_maturity : stream.shape[1] - (labelNumber_maturity-1) ].values
    labels_frequency    = stream.iloc[:, stream.shape[1] - labelNumber_frequency : stream.shape[1] - (labelNumber_frequency-1) ].values
    labels_coupon       = stream.iloc[:, stream.shape[1] - labelNumber_coupon : stream.shape[1] - (labelNumber_coupon-1) ].values
                                      
    # create one hot vector for labels
    def oneHot(length, n_classes, lab):
        a = np.zeros((features.shape[0], n_classes)) 
        for i, number in enumerate(lab):
            a[i][ int(number[0]) ] = 1        
        return a
            
    labels_maturity = oneHot(size, n_classes_maturity, labels_maturity)
    labels_frequency = oneHot(size, n_classes_frequency, labels_frequency)
    labels_coupon = oneHot(size, n_classes_coupon, labels_coupon)

    maximum                         = features[:,0]
    maximumPosition                 = features[:,1]
    entryPoint                      = features[:,2]
    deltaMaxFist                    = features[:,3]
    delta                           = features[:,4:363]
    jumps                           = features[:,363:722]
    timeAfterJump                   = features[:,722:1081]
    numberOfZeros                   = features[:,1081]
    maximalDistanceToTheNextDrop    = features[:,1082]
    mean                            = features[:,1083]
    variance                        = features[:,1084]
    numberOfJumps                   = features[:,1085]
    averageJump                     = features[:,1086]
    numberOfJumpsRelativeToLength   = features[:,1087]
    maximalJump                     = features[:,1088]
    curvature                       = features[:,1089:1447]

    probabilities_Maturity          = probs[:,:4]
    probabilities_Frequency         = probs[:,4:7]
    probabilities_Coupon            = probs[:,7:13]

    return maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, probabilities_Maturity, probabilities_Frequency, probabilities_Coupon

def oneHot_to_number(lab):
    a = np.zeros(lab.shape[0]) 
    for k, i in enumerate(lab):
        a[k] = np.argmax(i)
    return a

batch_size = 5000
 
reader = pd.read_csv(path_Features, header=None, sep=";", chunksize=batch_size, iterator=True)

maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, labels_maturity, labels_frequency, labels_coupon, _, _, _ = next_batch_ofFeatures(reader, batch_size)
labels_maturity = oneHot_to_number(labels_maturity)
labels_frequency = oneHot_to_number(labels_frequency)
labels_coupon = oneHot_to_number(labels_coupon)

l = {
     "maximum": maximum, 
     "maximumPosition": maximumPosition, 
     "entryPoint": entryPoint, 
     "deltaMaxFist": deltaMaxFist,
     "numberOfZeros": numberOfZeros,
     "maximalDistanceToTheNextDrop": maximalDistanceToTheNextDrop,
     "mean": mean,
     "variance": variance,
     "numberOfJumps": numberOfJumps,
     "averageJump": averageJump,
     "numberOfJumpsRelativeToLength": numberOfJumpsRelativeToLength,
     "maximalJump": maximalJump
    }

number = {
     "maximum": 0, 
     "maximumPosition": 1, 
     "entryPoint": 2, 
     "deltaMaxFist": 3,
     "numberOfZeros": 4,
     "maximalDistanceToTheNextDrop": 5,
     "mean": 6,
     "variance": 7,
     "numberOfJumps": 8,
     "averageJump": 9,
     "numberOfJumpsRelativeToLength": 10,
     "maximalJump": 11
    }

#######################
a = labels_coupon
b = "Coupon"
#######################
for index1, elem1 in enumerate(l):
    for index_y, elem2 in enumerate([el for index2, el in enumerate(l) if index2 > index1]):
        feature_X = l[elem1]
        feature_Y = l[elem2]
        
        feature_X_0 = feature_X[[i  for i,k in enumerate(a) if k ==0]]
        feature_X_1 = feature_X[[i  for i,k in enumerate(a) if k ==1]]
        feature_X_2 = feature_X[[i  for i,k in enumerate(a) if k ==2]]
        feature_X_3 = feature_X[[i  for i,k in enumerate(a) if k ==3]]
        feature_X_4 = feature_X[[i  for i,k in enumerate(a) if k ==4]]
        feature_X_5 = feature_X[[i  for i,k in enumerate(a) if k ==5]]
        
        feature_Y_0 = feature_Y[[i  for i,k in enumerate(a) if k ==0]]
        feature_Y_1 = feature_Y[[i  for i,k in enumerate(a) if k ==1]]
        feature_Y_2 = feature_Y[[i  for i,k in enumerate(a) if k ==2]]
        feature_Y_3 = feature_Y[[i  for i,k in enumerate(a) if k ==3]]
        feature_Y_4 = feature_Y[[i  for i,k in enumerate(a) if k ==4]]
        feature_Y_5 = feature_Y[[i  for i,k in enumerate(a) if k ==5]]
        
        plt.figure()
        plt.plot(feature_X_0, feature_Y_0, '.', color = 'c')
        plt.plot(feature_X_1, feature_Y_1, '.', color = 'm')
        plt.plot(feature_X_2, feature_Y_2, '.', color = 'r')
        plt.plot(feature_X_3, feature_Y_3, '.', color = 'k')
        plt.plot(feature_X_4, feature_Y_4, '.', color = 'b')
        plt.plot(feature_X_5, feature_Y_5, '.', color = 'g')
        plt.xlabel(elem1)
        plt.ylabel(elem2)
        plt.title(b + " Classes")
        
        plt.savefig(path_save + 'FeaturePlot-' + b + '-' + str(number[elem1]) + '-' + str(number[elem2]) + '.png')


"""
good plots:
    Coupon:
        maximum, numberOfZeros
        maximalJump, numberOfJumps
        maximalJump, ...
        entryPoint, numberOfJumps
        variance, maximalJump
    Maturity
        numberOfZeros, maximalJump
        numberOfZeros, maximum
        numberOfZeros, ...
    Frequency
        numberOfJumpsRelativeToLength, ...
        averageJump, numberOfJumpsRelativeToLength
        entryPoint, averageJump
        numberOfJumps, numberOfZeros
"""

