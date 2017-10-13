#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:46:28 2017

@author: Chris
"""
import numpy as np

def featureCalculator_np(data, timeSeriesLength, numberOfProfiles):
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

    ## Feature 16: curvature (second derivative)
    print("calclate features 16")
    curvature = np.zeros((numberOfProfiles, 358), dtype = np.float32)
    for i in range(0, numberOfProfiles):   #Loop over the input serieses    
        for j in range(0, timeSeriesLength - 2): #Loop over first derivative
            curvature[i,j] = delta[i,j] - delta[i,j+ 1]
        

        
    print("Features calculated")
    
    return maximum, maximumPosition, entryPoint, deltaMaxFist, delta, jumps, timeAfterJump, numberOfZeros, maximalDistanceToTheNextDrop, mean, variance, numberOfJumps, averageJump, numberOfJumpsRelativeToLength, maximalJump, curvature
