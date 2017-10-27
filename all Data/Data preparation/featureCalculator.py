#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:46:28 2017

@author: Chris

here I use cleaned and randomized data from "clean-randomize-data.py"
No NaN values, zero series or serieses with wrong length allowed.
The code calculates 16 features and saves them als csv.

"""
import numpy as np
import pandas as pd


def calculate_Features(path_SourceData, path_output = "leer"):
    if (path_output == "leer"):
        path_output = path_SourceData[:-4] + '-features.csv'

    completeData = pd.read_csv(path_SourceData, header=None, sep=";")
    data =      completeData.iloc[:, : completeData.shape[1] - 10 ].values
    lables =    completeData.iloc[:, completeData.shape[1] - 10 : ].values
    print("data loaded")
    
    timeSeriesLength = 360
    numberOfProfiles = data.shape[0]
    
    ## Feature 1 & 2: Maximum & Position of the maximum
    print("calclate features 1 & 2")
    maximum = np.zeros(numberOfProfiles, dtype = np.float32)
    maximumPosition = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses
        maximum[i] = max(data[i])
        negmaximum = max(-data[i])
        if(negmaximum > maximum[i]): 
            maximum[i] = negmaximum
            maximumPosition[i] = np.argmax(-data[i])
        else:
            maximumPosition[i] = np.argmax(data[i])
            
    ## Feature 3: First Point
    print("calclate features 3")
    entryPoint = data[:,0]
    
    ## Feature 4: Delta(maximum, first point)
    print("calclate features 4")
    deltaMaxFist = maximum - entryPoint
    
    ## Feature 5: neg delta to the next point (first derivative)
    print("calclate features 5")
    delta = np.diff(-data, n=1, axis=1)
    
    ## Feature 6: max(delta, 0) = down jumps
    print("calclate features 6")
    jumps = np.maximum(delta,0)        
            
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
    for i in range(numberOfProfiles):   #Loop over the input serieses    
        for k in range(timeSeriesLength-1,-1,-1):
            if data[i,k] == 0: numberOfZeros[i] += 1
    
    ## Feature 9: maximal distance to the next drop3
    print("calclate features 9")
    maximalDistanceToTheNextDrop = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses   
        maximalDistanceToTheNextDrop[i] = max(timeAfterJump[i,int(maximumPosition[i]):int(360-numberOfZeros[i])])
        
    ## Feature 10: mean of the tensor without last zeros
    print("calclate features 10")
    mean = np.mean(data, axis = 1)
        
    ## Feature 11: variance of the tensor without last zeros
    print("calclate features 11")
    variance = np.var(data, axis = 1)  
    
    ## Feature 12 & 13: number of jumps and average jump higth
    print("calclate features 12 & 13")
    numberOfJumps = np.zeros(numberOfProfiles, dtype = np.float32)
    averageJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfJumps[i] = (jumps[i] > 0).sum()
        listOfJumps = [jump for jump in jumps[i] if jump > 0]
        if listOfJumps == []:
            averageJump[i] = 0
        else:
            averageJump[i] = np.mean(listOfJumps)
    
    ## Feature 14: number of jumps relative to net (real) time series length
    print("calclate features 14")
    numberOfJumpsRelativeToLength = numberOfJumps / (360-numberOfZeros)
    
    ## Feature 15: highest jump
    print("calclate features 15")
    maximalJump = np.max(jumps, axis = 1)
    
    ## Feature 16: curvature (second derivative)
    print("calclate features 16")
    curvature = np.diff(delta, n=1, axis=1)
    
    print("Features calculated")
    
    concat = np.column_stack((maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, delta, jumps, timeAfterJump, curvature, lables))
                              
    print("data stacked together") # inklisive lables
    
    df = pd.DataFrame(concat)
    df.to_csv(path_output, header=None, sep=";", index = False, index_label = False)
    print("data saved")
    
    del completeData, data
    del maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature, lables


def calculate_Features_return(data, timeSeriesLength, numberOfProfiles):

    ## Feature 1 & 2: Maximum & Position of the maximum
    print("calclate features 1 & 2")
    maximum = np.zeros(numberOfProfiles, dtype = np.float32)
    maximumPosition = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses
        maximum[i] = max(data[i])
        negmaximum = max(-data[i])
        if(negmaximum > maximum[i]): 
            maximum[i] = negmaximum
            maximumPosition[i] = np.argmax(-data[i])
        else:
            maximumPosition[i] = np.argmax(data[i])
            
    ## Feature 3: First Point
    print("calclate features 3")
    entryPoint = data[:,0]
    
    ## Feature 4: Delta(maximum, first point)
    print("calclate features 4")
    deltaMaxFist = maximum - entryPoint
    
    ## Feature 5: neg delta to the next point (first derivative)
    print("calclate features 5")
    delta = np.diff(-data, n=1, axis=1)
    
    ## Feature 6: max(delta, 0) = down jumps
    print("calclate features 6")
    jumps = np.maximum(delta,0)        
            
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
    for i in range(numberOfProfiles):   #Loop over the input serieses    
        for k in range(timeSeriesLength-1,-1,-1):
            if data[i,k] == 0: numberOfZeros[i] += 1
    
    ## Feature 9: maximal distance to the next drop3
    print("calclate features 9")
    maximalDistanceToTheNextDrop = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses   
        maximalDistanceToTheNextDrop[i] = max(timeAfterJump[i,int(maximumPosition[i]):int(360-numberOfZeros[i])])
        
    ## Feature 10: mean of the tensor without last zeros
    print("calclate features 10")
    mean = np.mean(data, axis = 1)
        
    ## Feature 11: variance of the tensor without last zeros
    print("calclate features 11")
    variance = np.var(data, axis = 1)  
    
    ## Feature 12 & 13: number of jumps and average jump higth
    print("calclate features 12 & 13")
    numberOfJumps = np.zeros(numberOfProfiles, dtype = np.float32)
    averageJump = np.zeros(numberOfProfiles, dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        numberOfJumps[i] = (jumps[i] > 0).sum()
        listOfJumps = [jump for jump in jumps[i] if jump > 0]
        if listOfJumps == []:
            averageJump[i] = 0
        else:
            averageJump[i] = np.mean(listOfJumps)
    
    ## Feature 14: number of jumps relative to net (real) time series length
    print("calclate features 14")
    numberOfJumpsRelativeToLength = numberOfJumps / (360-numberOfZeros)
    
    ## Feature 15: highest jump
    print("calclate features 15")
    maximalJump = np.max(jumps, axis = 1)
    
    ## Feature 16: curvature (second derivative)
    print("calclate features 16")
    curvature = np.diff(delta, n=1, axis=1)
    
    print("Features calculated")
    
    oneDimFeatures = np.column_stack((maximum, maximumPosition, entryPoint, deltaMaxFist, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump))

    return oneDimFeatures, delta, jumps, timeAfterJump, curvature
