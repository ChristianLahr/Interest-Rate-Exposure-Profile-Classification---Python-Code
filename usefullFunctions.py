#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:59:13 2017

@author: Chris
"""

import matplotlib.pyplot as plt


def label_extraction_prams(label):
    if(label=="maturity"):
        labelNumber = 9
        n_classes = 4
    elif(label=="coupon"):
        labelNumber = 5
        n_classes = 6        
    elif(label=="frequency"):
        labelNumber = 7
        n_classes = 3
    elif(label=="difference-curvelevel-coupon"):
        labelNumber = 1
        n_classes = 12
    else:
        labelNumber = -1
        n_classes = -1
    return labelNumber, n_classes

def plotResults(iterations, costs, accuracys, training_iters, accuracys_dev = None, withDevEvaluation = False, testAccuracy = 0.1, withtestAccuracy = False):
    fig = plt.figure(figsize=(10,4), dpi=60)
    fig.add_subplot(1, 2, 1)
    plt.plot(iterations, costs, label="Cost")
    plt.title('Cost evolution')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.xlim((0, training_iters))
    plt.ylim((0, costs[0]*1.1))
    plt.legend()
    
    fig.add_subplot(1, 2, 2)
    plt.plot(iterations, accuracys, label="Accuracy")
    plt.title('accuracy evolution')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.xlim((0, training_iters))
    plt.ylim((0, 1.1))
    if withDevEvaluation:
        plt.plot(iterations, accuracys_dev, label="Accuracy Dev")
    if withtestAccuracy:
        plt.plot([iterations[0], iterations[-1]], [testAccuracy, testAccuracy], label="Test accuracy")
    plt.legend()
                
    plt.show()                

def plotResults_4Accuracies(iterations, costs, accuracys_all, accuracys_Maturity, accuracys_Frequency, accuracys_Coupon, training_iters, accuracys_all_dev = None, accuracys_Maturity_dev = None, accuracys_Frequency_dev = None, accuracys_Coupon_dev = None, withDevEvaluation = False, testAccuracy = 0.1, withtestAccuracy = False):
    fig = plt.figure(figsize=(10,4), dpi=60)
    fig.add_subplot(1, 2, 1)
    plt.plot(iterations, costs, label="Cost")
    plt.title('Cost evolution')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.xlim((0, training_iters))
    plt.ylim((0, costs[0]*1.1))
    plt.legend()
    
    fig.add_subplot(1, 2, 2)
    plt.plot(iterations, accuracys_all, label="Accuracy all")
    plt.plot(iterations, accuracys_Maturity, label="Accuracy Maturity")
    plt.plot(iterations, accuracys_Frequency, label="Accuracy Frequency")
    plt.plot(iterations, accuracys_Coupon, label="Accuracy Coupon")
    plt.title('accuracy evolution')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.xlim((0, training_iters))
    plt.ylim((0, 1.1))
    if withDevEvaluation:                                                                                                                        
        plt.plot(iterations, accuracys_all_dev, label="Accuracy Dev all")
        plt.plot(iterations, accuracys_Maturity_dev, label="Accuracy Dev Maturity")
        plt.plot(iterations, accuracys_Frequency_dev, label="Accuracy Dev Frequency")
        plt.plot(iterations, accuracys_Coupon_dev, label="Accuracy Dev Coupon")
    if withtestAccuracy:
        plt.plot([iterations[0], iterations[-1]], [testAccuracy, testAccuracy], label="Test accuracy")
    plt.legend()
                
    plt.show() 