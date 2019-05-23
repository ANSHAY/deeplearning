# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:17:26 2018

@author: ANSHAY
"""

import numpy as np
import h5py

def loadCatData(path):
    """
    loads data from h5 file for cat vs non cat 
    path- path for the folder containing data files
    returns:
        training and test data and list of classes
        trainDataX, trainDataY, testDataX, testDataY, classes
        size is (num_features x num_samples)
    """
    trainData = h5py.File(path + '/train_catvnoncat.h5', "r")
    trainDataX = np.array(trainData["train_set_x"][:]) # your train set features
    trainDataY = np.array(trainData["train_set_y"][:]) # your train set labels

    testData = h5py.File(path + '/test_catvnoncat.h5', "r")
    testDataX = np.array(testData["test_set_x"][:]) # your test set features
    testDataY = np.array(testData["test_set_y"][:]) # your test set labels

    classes = np.array(testData["list_classes"][:]) # the list of classes
    trainDataY = trainDataY.reshape((1, trainDataY.shape[0]))
    testDataY = testDataY.reshape((1, testDataY.shape[0]))
    
    return trainDataX, trainDataY, testDataX, testDataY, classes

def loadPlanarData():
    """
    defines randon planar data for 2 classes red and blue
    returns:
        data, labels and list of classes
        X, Y, classes
        size is (num_features x num_samples)
    """
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    classes = ['red', 'blue']
    X = X.T
    Y = Y.T
    return X, Y, classes
