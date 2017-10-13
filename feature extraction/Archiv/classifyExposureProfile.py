#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:11:09 2017

@author: Chris
"""
import numpy as np
from featureCalculator_np import featureCalculator_np
from usefullFunctions import loadDataFromCSVunlabeled
from useTrainedModel_fct import useTrainedModel, useTrainedModel_DeepNetwork


### Classification with Model_deepNetwork_shorter

withLabels = False

### Load Exposure Profiles
#CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/Feature Extraction Test Data/Exposure_0.05Quantile_04062017 zusammengesetzt-unvollständig-clean.csv'
CSV_Name = '/Users/Chris/Python/Machine Learning/Masterarbeit1/all Data/O5887085_interpolate_english_normalizedNotional.csv'

if(withLabels):
    data_e = loadDataFromCSVunlabeled(CSV_Name, False, 10)
else:
    data_e = loadDataFromCSVunlabeled(CSV_Name, False, 0)

data_e = data_e[:, :]

### clear the data from zero lines
zeroLine = np.zeros(360)
badIndexes = []
for index, item in enumerate(data_e):
#    if(index==0): print(item)
    if(np.array_equal(item, zeroLine)): badIndexes.append(index)
for i in reversed(badIndexes):
    data_e = np.delete(data_e, i, 0)

labelNumber_maturity = 9
labelNumber_frequency = 7
labelNumber_coupon = 5

n_classes_maturity = 4
n_classes_frequency = 3
n_classes_coupon = 6

timeSeriesLength_e = data_e.shape[1]
numberOfProfiles_e = data_e.shape[0]

### calculate the features and store them in the rigth way
maximum_e, maximumPosition_e, entryPoint_e, deltaMaxFist_e, delta_e, jumps_e, timeAfterJump_e, numberOfZeros_e, maximalDistanceToTheNextDrop_e, mean_e, variance_e, numberOfJumps_e, averageJump_e, numberOfJumpsRelativeToLength_e, maximalJump_e = featureCalculator_np(data_e, timeSeriesLength_e, numberOfProfiles_e)

   
   
### load Data from trained models as extra input for deeper model
print("Calculate trained model outputs to get more information for the deep network")
networkInput = [maximum_e, maximumPosition_e, entryPoint_e, numberOfZeros_e, mean_e, variance_e, maximalDistanceToTheNextDrop_e, deltaMaxFist_e, numberOfJumpsRelativeToLength_e, maximalJump_e, jumps_e]
numberWeigthsPerNeuronInFirstLayer_usedToTrainTheSmalModels = 200
name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Maturity.ckpt'
probabilities_Maturity_e = useTrainedModel(networkInput, name_SavesModel, n_classes_maturity, numberWeigthsPerNeuronInFirstLayer_usedToTrainTheSmalModels)
name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Frequency.ckpt'
probabilities_Frequency_e = useTrainedModel(networkInput, name_SavesModel, n_classes_frequency, numberWeigthsPerNeuronInFirstLayer_usedToTrainTheSmalModels)
name_SavesModel = 'ownFeatureExtraction_MultiDimInput_Coupon.ckpt'
probabilities_Coupon_e = useTrainedModel(networkInput, name_SavesModel, n_classes_coupon, numberWeigthsPerNeuronInFirstLayer_usedToTrainTheSmalModels)
print("Additional inputs calculated")
 
numberWeigthsPerNeuronInFirstLayer_usedToTrainTheModel = 200
name_SavesModel = 'ownFeatureExtraction_DeepNetwork_shorter_test.ckpt'
if(withLabels):   
    None
    #    networkInput = [maximum_e, maximumPosition_e, entryPoint_e, numberOfZeros_e, mean_e, variance_e, maximalDistanceToTheNextDrop_e, deltaMaxFist_e, numberOfJumpsRelativeToLength_e, maximalJump_e, jumps_e, probabilities_Maturity_e, probabilities_Frequency_e, probabilities_Coupon_e, labels_maturity_e, labels_frequency_e, labels_coupon_e]
    # classes_Maturity, classes_Frequency, classes_Coupon, correct_preded_Maturity, correct_preded_Frequency, correct_preded_Coupon, accuracy_Maturity, accuracy_Frequency, accuracy_Coupon, accuracy_all = useTrainedModel_DeepNetwork(networkInput, name_SavesModel, numberWeigthsPerNeuronInFirstLayer_usedToTrainTheModel, withLabels)        
else:
    networkInput = [maximum_e, maximumPosition_e, entryPoint_e, numberOfZeros_e, mean_e, variance_e, maximalDistanceToTheNextDrop_e, deltaMaxFist_e, numberOfJumpsRelativeToLength_e, maximalJump_e, jumps_e, probabilities_Maturity_e, probabilities_Frequency_e, probabilities_Coupon_e]
    classes_Maturity, classes_Frequency, classes_Coupon = useTrainedModel_DeepNetwork(networkInput, name_SavesModel, numberWeigthsPerNeuronInFirstLayer_usedToTrainTheModel, withLabels)

def maturityMap(x):
    return {
        0: '[0, 5]',
        1: '(5, 10]',
        2: '(10, 20]',
        3: '(20, 30]'
        }.get(x, 'error')

def frequencyMap(x):
    return {
        0: 'quarterly',
        1: 'semiannual',
        2: 'annual'
        }.get(x, 'error')

def couponMap(x):
    return {
        0: '[0, 0.005]',
        1: '(0.005, 0.01]',
        2: '(0.01, 0.02]',
        3: '(0.02, 0.04]',
        4: '(0.04, 0.08]',
        5: ' > 0.08'
        }.get(x, 'error')

for i in range(0, numberOfProfiles_e):
    maturity = maturityMap(classes_Maturity[i])
    frequency = frequencyMap(classes_Frequency[i])
    coupon = couponMap(classes_Coupon[i])
    print(maturity, "years, ", frequency, "payment, coupon in ", coupon)

"""    
[0-5]; (5-10]; (10-20]; (20-30]
Quarterly; Semiannual; Annual
[0, 0.005]; (0.005, 0.01], (0.01, 0.02], (0.02, 0.04], (0.04, 0.08], (0.008, ...]
"""