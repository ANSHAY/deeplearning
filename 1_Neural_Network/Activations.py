# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 19:09:06 2018

This Module contains the following activation functions as well as their 
backward propagation functions, to be used in a neural network
relu
tanh
sigmoid
binary

@author: ANSHAY
"""

import numpy as np

def relu(Z):
    """
    computes relu activation for a given matrix
    """
    A = np.maximum(Z, 0)
    return A

def tanh(Z):
    """
    computes tanh activation for a given matrix
    """
    A = np.tanh(Z)
    return A

def sigmoid(Z):
    """
    computes sigmoid activation for a given matrix
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    return A

def binary(Z):
    """
    computes binary activation for a given matrix
    """
    A = Z > 0.5
    return A

def relu_back(Z, dA):
    """
    computes relu back activation for a given matrix
    """
    dZ = np.array(dA, copy=True)
    dZ[Z<0.0] = 0.0
    return dZ

def tanh_back(Z, dA):
    """
    computes tanh activation for a given matrix
    """
    A = np.tanh(Z)
    dZ = dA * (1 - np.power(A, 2))
    return dZ

def sigmoid_back(Z, dA):
    """
    computes sigmoid activation for a given matrix
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * A * (1 - A)
    return dZ

def binary_back(Z, dA):
    """
    computes binary activation for a given matrix
    """
    dZ = np.array(dA, copy=True)
    dZ[Z < 0.0] = 0.0
    dZ[Z > 0.0] = 0.0
    return dZ