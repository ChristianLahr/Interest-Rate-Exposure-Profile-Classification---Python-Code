#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:07:18 2017

@author: Chris
"""

import os
from clean_randomize_data import cleanANDRandomize
from featureCalculator_probsFromTrainedModel import calculateFeaturesANDProbsFromTrainedModel
from separate_Train_Dev_Test_Sets import separate_train_dev_test_Sets
from featureCalculator import calculate_Features

featureCalculation = True
featureCalculationWithProbs = False

cwd = os.getcwd()
path_sourceData             = cwd + '/../TestBox/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig.csv'
path_sourceData_clean_rand  = path_sourceData[:-4] +  '-clean-rand.csv'

## clean from buggy data sets, randomize and store them 
#cleanANDRandomize(path_sourceData, path_sourceData_clean_rand)

# calculate features and probabilities from trained models --> deep network input
if featureCalculationWithProbs:
    path_DeepNetworkInput = path_sourceData_clean_rand[:-4] + '_DeepNetworkInput.csv'
    calculateFeaturesANDProbsFromTrainedModel(path_sourceData_clean_rand, path_DeepNetworkInput)

# calculate featres and store them
if featureCalculation:
    path_features_input = path_sourceData_clean_rand
    path_features_output = path_features_input[:-4] + '-features.csv'
    calculate_Features(path_features_input, path_features_output)

# separate train, dev, and test sets and save under the right path/name
if featureCalculation:
    separate_train_dev_test_Sets(path_features_output)
elif(featureCalculationWithProbs):
    separate_train_dev_test_Sets(path_DeepNetworkInput)
else:
    separate_train_dev_test_Sets(path_sourceData_clean_rand)
print("Data prepared")



