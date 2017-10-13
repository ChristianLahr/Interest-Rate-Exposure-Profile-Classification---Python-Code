#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:59:13 2017

@author: Chris
"""

import pandas as pd
import numpy as np

def randomizeRows(VectData, VectLabel):
    # brings the rows of both vectors in the same new random order
    # both vectors need same length
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return tempData, tempLabel

def randomizeRowsOfArraysInList(liste, VectLabel):
    # brings the rows of all vectors of the list and the labels vector in the same new random order
    # all vectors need same length
    indices = np.random.permutation(len(liste[0]))
    templist = liste
    for j in range(0, len(liste)):
        templist[j] = [liste[j][i] for i in indices]
    tempLabel = [VectLabel[i] for i in indices]
    return templist, tempLabel
    
def randomizeRowsUnlabeled(VectData):
    indices = np.random.permutation(len(VectData))
    tempData = [VectData[i] for i in indices]
    return tempData

def loadDataFromCSV(path, normalize, labelNumber, n_classes): # normalize = True --> normalization over altitude of every row (=time series)
    randomize = False
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
            # also normalizes negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            if(maximum == 0): maximum = 1 #if it is a zero line we need no normalization (avoid deviding througth zero)
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
    

def loadDataFromCSVrealNumber(path, labelNumber):
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    ###### select here the labels to be analyzed
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    labels = completeData.iloc[:, completeData.shape[1] - labelNumber : completeData.shape[1] - (labelNumber-1) ].values

    print('Data loaded')
    return datapoints, labels    
    
    
def loadDataFromCSVunlabeled(path, normalize, offset): # normalize = True --> normalization over altitude of every row (=time series)
    # Import other test data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:, : completeData.shape[1] - offset ].values
    
    if normalize == True:
        for i in range(0, datapoints.shape[0]):
            # also normalize negative values!!
            maximum = max( np.ndarray.max(datapoints[i]), -np.ndarray.min(datapoints[i]))
            for j in range(0, datapoints.shape[1]):
                datapoints[i][j] = (datapoints[i][j] / maximum)

    datapoints_rand = np.float32(datapoints)
        
    print('Data without labels loaded')
    return datapoints_rand
    
def loadAllLabelsFromCSV(path): # loads maturity frewuency and coupon labels
    # Import data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    numberOfProfiles = datapoints.shape[0]

    labelNumber_maturity = 9
    labelNumber_frequency = 7
    labelNumber_coupon = 5

    n_classes_maturity = 4
    n_classes_frequency = 3
    n_classes_coupon = 6
    
    labels_maturity = completeData.iloc[:, completeData.shape[1] - labelNumber_maturity : completeData.shape[1] - (labelNumber_maturity-1) ].values
    labels_frequency = completeData.iloc[:, completeData.shape[1] - labelNumber_frequency : completeData.shape[1] - (labelNumber_frequency-1) ].values
    labels_coupon = completeData.iloc[:, completeData.shape[1] - labelNumber_coupon : completeData.shape[1] - (labelNumber_coupon-1) ].values    
    
    # consturct the one hot vectors
    a = np.zeros((numberOfProfiles, n_classes_maturity)) 
    for i, number in enumerate(labels_maturity):
        a[i][ int(number[0]) ] = 1        
    labels_Matrix = a    
    labels_maturity_matrix = np.float32(labels_Matrix)
    
    a = np.zeros((numberOfProfiles, n_classes_frequency)) 
    for i, number in enumerate(labels_frequency):
        a[i][ int(number[0]) ] = 1        
    labels_Matrix = a    
    labels_frequency_matrix = np.float32(labels_Matrix)

    a = np.zeros((numberOfProfiles, n_classes_coupon)) 
    for i, number in enumerate(labels_coupon):
        a[i][ int(number[0]) ] = 1        
    labels_Matrix = a    
    labels_coupon_matrix = np.float32(labels_Matrix)
    
    datapoints_return = np.float32(datapoints)

    print('Data loaded')
    return datapoints_return, labels_maturity_matrix, labels_frequency_matrix, labels_coupon_matrix
    
    
def loadAllLabelsFromCSV_72classes(path): # normalize = True --> normalization over altitude of every row (=time series)
    # Import data
    completeData = pd.read_csv(path, header=None, sep=";")
    datapoints = completeData.iloc[:, : completeData.shape[1] - 10 ].values
    numberOfProfiles = datapoints.shape[0]

    labelNumber_maturity = 9
    labelNumber_frequency = 7
    labelNumber_coupon = 5

    n_classes_maturity = 4
    n_classes_frequency = 3
    n_classes_coupon = 6
    
    labels_maturity = completeData.iloc[:, completeData.shape[1] - labelNumber_maturity : completeData.shape[1] - (labelNumber_maturity-1) ].values
    labels_frequency = completeData.iloc[:, completeData.shape[1] - labelNumber_frequency : completeData.shape[1] - (labelNumber_frequency-1) ].values
    labels_coupon = completeData.iloc[:, completeData.shape[1] - labelNumber_coupon : completeData.shape[1] - (labelNumber_coupon-1) ].values    
    
    # consturct the one hot vectors
    a = np.zeros((numberOfProfiles, n_classes_maturity * n_classes_frequency * n_classes_coupon)) 
    for i, number_mat in enumerate(labels_maturity):
        number_frequ = labels_frequency[i]
        number_coup = labels_coupon[i]
        a[i][ int(    (number_mat + n_classes_maturity*number_frequ) + n_classes_maturity * n_classes_frequency * number_coup    ) ] = 1        
    labels_72_matrix = np.float32(a)    
        
    datapoints_return = np.float32(datapoints)

    print('Data loaded')
    return datapoints_return, labels_72_matrix

    
    
    
    
    
    
    