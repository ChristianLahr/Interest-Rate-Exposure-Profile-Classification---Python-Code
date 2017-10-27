#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:32:19 2017

@author: Chris
"""

import pandas as pd
import numpy as np
import math

def cleanANDRandomize(fileName, fileName_new):
#    fileName =      '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
#    fileName_new =  '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand.csv'
    
    completeData = pd.read_csv(fileName, header=None, sep=";")
    data = completeData.iloc[:,:].values
    print("data loaded")
    
    # check length and delete
    indexLength = []
    for i, row in enumerate(data):
        if(len(row) != 370 ): # or type(row[k] is str)
            indexLength.append(i)
    data = np.delete(data,indexLength, axis=0) 
    
    # check NaNs and delete
    indexNaNs = []
    for i, row in enumerate(data):
        if len([k for k in row[0:360] if math.isnan(k)]) != 0: 
            indexNaNs.append(i)
    data = np.delete(data,indexNaNs, axis=0) 
    
    # check zero serieses and delete
    indexZeros = []
    for i, row in enumerate(data):
        if len([k for k in row[0:360] if k != 0]) == 0:
            indexZeros.append(i)
    data = np.delete(data,indexZeros, axis=0) 
    
    print(len(indexNaNs) + len(indexLength) + len(indexZeros))
    print(len(indexZeros))
    print("data cleaned")
    
    indices = np.random.permutation(len(data))
    data_randomized = [data[i] for i in indices]
    data_randomized = data_randomized
    print("data randomized")
    
    df = pd.DataFrame(data_randomized)
    df.to_csv(fileName_new, header=None, sep=";", index = False, index_label = False)
    print("data saved")
    
    del completeData
    del data
    del indices
    del data_randomized
    del df 
    del row 
    print("memory free")
