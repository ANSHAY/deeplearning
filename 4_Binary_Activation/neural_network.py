# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:51:51 2018

@author: ANSHAY
"""
import numpy as np
import Activations
import matplotlib.pyplot as plt

def initialize_parameters(layer_sizes, init_type='none'):
    """
    initializes Weight and Bias matrices for the neural network
    for given number of layer sizes
    inputs:
        layer_sizes : array of input_size and layer sizes such that
                  layer_sizes[0] contains input size
        inti_types: type of initialization.. he, Xavier,                   
    """
    print ('\nInitializing Parameters..............')
    parameters = {}
    for l in range(1, len(layer_sizes)):
        if init_type == 'he':
            init_factor = np.sqrt(2 / layer_sizes[l])
        elif init_type == 'xavier':
            init_factor = np.sqrt(2 / (layer_sizes[l] + layer_sizes[l-1]))
        else:
            init_factor = 1.0    
        parameters['W'+str(l)] = np.random.randn(layer_sizes[l], \
                                                 layer_sizes[l-1]) * init_factor
        parameters['B'+str(l)] = np.zeros((layer_sizes[l],1))
    return parameters

def activation(Z, activation_type):
    """
    calls the appropriate activation function
    """
    if activation_type == 'relu':
        return Activations.relu(Z)
    if activation_type == 'tanh':
        return Activations.tanh(Z)
    if activation_type == 'sigmoid':
        return Activations.sigmoid(Z)
    if activation_type == 'binary':
        return Activations.binary(Z)
    else:
        return Activations.relu(Z)        

def compute_cost(y, y_hat, parameters, lamda):
    """
    computes the overall cost for the data corresponding to current 
    parameters and activations
                cost = sum(y * log(y_hat) + (1-y) * log(1-y_hat))
    input:
        y: correct/desired output for every data sample. size (1 x num_samples)
        y_hat: final activation of the network i.e. output of the neural network
            size --- (1 x num_samples)
        parameters: weight and bias for regularization
        lamda: regularization parameter. 0 means regularization not applied
    returns:
        cost
    """
    m = float(len(y_hat))
#    regularization = np.sum([np.sum(np.square(parameters['W'+str(i+1)]))\
#                             for i in range(len(parameters)//2)])
#    regularization *= lamda/(2.0)
#    cost = -((np.dot(y, np.log(y_hat).T) + np.dot((1-y), np.log(1-y_hat).T)) + regularization) / m
    cost = np.sum((y != y_hat)) / m
    return cost

def forward_propagate(data, parameters, metadata):
    """
    computes forward propagation on data and computes caches- Z and activation A
    inputs:
        data: size -- (num_features x num_samples),
        parameters: size 2* num_layers, containing Weight W and Bias B for
                    every layer
        metadata: contains-
            hidden_activation- type of activation function for hidden layers
            final_activation- type of activation function for final layer
            lamda- for cost computation
    returns:
        caches containing activations and Z for every layer
    """
    caches = {'A0':data}
    num_layers = len(parameters)//2
    for l in range(1, num_layers):
        # calculate Z = W*X + B
        Z_l = np.dot(parameters['W'+str(l)], caches['A'+str(l-1)]) + parameters['B'+str(l)]
        # calculate A = activtion function(Z)
        A_l = activation(Z_l, metadata['hidden_activation'])
        # put Z and A in caches
        caches['Z'+str(l)] = Z_l
        caches['A'+str(l)] = A_l
    # for final layer
    Z_L = np.dot(parameters['W'+str(num_layers)], caches['A'+str(num_layers-1)])\
                 + parameters['B'+str(num_layers)]
    A_L = activation(Z_L, metadata['final_activation'])
    caches['Z'+str(num_layers)] = Z_L
    caches['A'+str(num_layers)] = A_L

    return caches

def activation_back(Z, dA, activation_type):
    """
    calls the appropriate backward activation function
    """
    if activation_type == 'relu':
        return Activations.relu_back(Z, dA)
    if activation_type == 'tanh':
        return Activations.tanh_back(Z, dA)
    if activation_type == 'sigmoid':
        return Activations.sigmoid_back(Z, dA)
    if activation_type == 'binary':
        return Activations.binary_back(Z, dA)
    else:
        return Activations.relu_back(Z, dA)

def gradients(dZ, A_1, W, lamda):
    m = float(dZ.shape[1])
    dW = (np.dot(dZ, A_1.T) + W*lamda) / m
    dB = np.mean(dZ, axis=-1, keepdims=True)
    dA_1 = np.dot(W.T, dZ)
    return dW, dB, dA_1

def backward_propagate(Y, parameters, caches, metadata):
    """
    computes gradients by backward propagation
    inputs:
        Y- correct labels
        parameters- weights and biases for all layers , W and B
        caches- contains activations A, and Z for every layer
        final_activation- activation type for final layer
        hidden_activation- activation type for hidden layers
    return:
        grads- gradients for weights and biases , dW, dB
    """
    num_layers = len(parameters)//2
    grads = {}
    m = Y.shape[1]
    
    Yl = Y
    for l in range(num_layers, 0, -1):
        Al = caches['A'+str(l)]
        Al_1 = caches['A'+str(l-1)]
        Wl = parameters['W'+str(l)]
        dWl = -np.dot(Yl, (1-Al_1).T) * Wl +\
              np.dot((2 * Yl - 1) * (2 * (Yl != Al) + 1), Al_1.T) * Wl
        nl = Wl.shape[0]
        nl_1 = Wl.shape[1]
        dBl = np.zeros((nl, 1))
        grads['dW'+str(l)] = dWl
        grads['dB'+str(l)] = dBl
        W = Wl>0
        ## Yl_1 = sum(Al == Wl>0).......... size-- (nl_1 * m)
        Yl = np.sum((Al.reshape((nl, 1, m))) == (W.reshape((nl, nl_1, 1))), axis=0)>0
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    updates the parameters based on the gradients computed through
    back propagation
    input:
        parameters- weights and biases for all layers , W and B
        grads- gradients for weights and biases , dW, dB
        learning_rate- rate at which gradient descent moves/updates
    returns:
        parameters- updated parameters
    """
    num_layers = len(parameters)//2
    for i in range(1, num_layers+1):
        W = parameters['W'+str(i)]
        W += learning_rate * grads['dW'+str(i)]
        W[W>1.0] = 1.0
        W[W<-1.0] = -1.0
        parameters['W'+str(i)] = W
        parameters['B'+str(i)] += learning_rate * grads['dB'+str(i)]
    
    return parameters

def train_model(train_X, train_Y, metadata):
    """
    trains and returns a neural network model for the given layer sizes,
    metadata, hyperparameters over the input data for the given number of
    iterations in metadata
    input:
        train_X: size -- (num_features x num_samples),
        train_Y: correct labels for training data. size- (1 x num_samples)
        metadata: contains-
            hidden_activation- type of activation function for hidden layers
            final_activation- type of activation function for final layer
            lamda- for cost computation
            iterations- number of iterations to train over
            init_type- type of initialization
            learning_rate- rate at which gradient descent moves/updates
            layer_sizes- array of size of each layer of neural network
    """
    print ('\nTraining model......................')
    layer_sizes = [train_X.shape[0]] + metadata['layer_sizes']
    iterations = metadata['iterations']
    parameters = initialize_parameters(layer_sizes, metadata['init_type'])
    num_layers = len(parameters)//2
    costs = []
    for i in range(iterations):
        caches = forward_propagate(train_X, parameters, metadata)
        y_hat = caches['A' + str(num_layers)]
        cost = compute_cost(train_Y, y_hat,\
                            parameters, metadata['lamda'])
        grads = backward_propagate(train_Y, parameters, caches, metadata)
        parameters = update_parameters(parameters, grads, metadata['learning_rate'])
        if i % 100 == 0:
            print ("Iteration " + str(i) + " Cost....................." + str(cost))
            costs.append(np.squeeze(cost))
    ## plot costs
    plt.plot(costs)
    model = {'parameters': parameters, 'caches': caches, 'cost': cost,\
             'metadata': metadata}
    print ('\nModel trained.............')
    return model

def predict(model, data):
    """
    predicts the classes on the given data for the given model
    """
    print ('\npredicting classes for the data.....')
    num_layers = len(model['parameters'])//2
    caches = forward_propagate(data, model['parameters'], model['metadata'])
    prediction = caches['A' + str(num_layers)] > 0.5
    return prediction

def test_model(model, test_X, test_Y):
    """
    computes accuracy for the model over test data
    returns:
        accuracy
    """
    print ('\nTesting model..........')
    y_hat = predict(model, test_X)
    accuracy = 1 - np.mean(np.abs(test_Y - y_hat))
    return accuracy