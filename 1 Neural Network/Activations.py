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
    A = np.max(Z, 0)
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
