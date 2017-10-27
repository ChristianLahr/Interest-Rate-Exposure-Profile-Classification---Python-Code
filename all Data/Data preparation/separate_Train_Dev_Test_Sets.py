#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:21:25 2017

@author: Chris
"""
import pandas as pd

def separate_train_dev_test_Sets(pathInput):
    completeData = pd.read_csv(pathInput, header=None, sep=";")
    numberOfTestSets = min(int(completeData.shape[0] * 0.1), 5000) 
    numberOfDevSets = numberOfTestSets
    
    train_set   = completeData.iloc[:-(numberOfTestSets+numberOfDevSets),:].values
    dev_set     = completeData.iloc[-(numberOfTestSets+numberOfDevSets):-numberOfTestSets,:].values
    test_set    = completeData.iloc[-numberOfTestSets:,:].values
    
    print("Shape train sets", train_set.shape)
    print("Shape dev sets", dev_set.shape)
    print("Shape test sets", test_set.shape)
    
    df = pd.DataFrame(train_set)
    df.to_csv(pathInput[:-4]+"_train.csv", header=None, sep=";", index = False, index_label = False)
    df = pd.DataFrame(dev_set)
    df.to_csv(pathInput[:-4]+"_dev.csv", header=None, sep=";", index = False, index_label = False)
    df = pd.DataFrame(test_set)
    df.to_csv(pathInput[:-4]+"_test.csv", header=None, sep=";", index = False, index_label = False)
    
    print("train, dev, test separated")
    
    