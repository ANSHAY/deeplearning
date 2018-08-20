# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:34:05 2018

This loads the data, defines hyperparameter, calls the neural network on train
data, tests the model on test data, prints accuracy, plots graphs and other
stuff


@author: ANSHAY
"""
import numpy as np
import pickle

def load_data():
    """
    loads the data for learning and separates into training and test data
    """
    
    return train_X, train_Y, test_X, test_Y, classes

def standardize(data, train_mean = 0, train_deviation = 1):
    """
    normalizes the data by subtracting mean and dividing by standard deviation
    for train data it computes the mean and deviation
    for test data it should use the training mean and deviation
    So, it also returns train_mean and train_deviation
    """
    if train_mean == 0:
        train_mean = np.mean(data)
    if train_deviation == 1:
        train_deviation = np.std(data)
    data = data - train_mean
    data = data / train_deviation
    return data, train_mean, train_deviation

def load_hyperparameters(filename):
    """
    loads metadata from external file named metadata
    """
    f = open(filename, 'r')
    metadata = pickle.load(f)
    f.close()
    return metadata

def train_model():
    return

def test_model():
    return

def print_all():
    return

def main():
    train_X, train_Y, test_X, test_Y, classes = load_data()
    train_X, train_mean, train_deviation = standardize(train_X)
    metadata = load_hyperparameters('metadata.pkl')
    metadata['train_mean'] = train_mean
    metadata['train_deviation'] = train_deviation
    
