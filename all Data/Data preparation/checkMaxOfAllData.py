#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:52:25 2017

@author: Chris
"""

import pandas as pd
import numpy as np
import os
cwd = os.getcwd()

def loadDataFromCSV(path): # normalize = True --> normalization over altitude of every row (=time series)
    completeData = pd.read_csv(path, header=None, sep=";")
    maxs = completeData.iloc[:, 0].values
    return maxs
    
file1Name = cwd + '/../Data from CIP/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean-rand-features.csv'
maxs = loadDataFromCSV(file1Name) # normalize?, randomize ? 
left = np.max(-maxs)
right = np.max(maxs)
print("[", left, ", ", right, "]")
