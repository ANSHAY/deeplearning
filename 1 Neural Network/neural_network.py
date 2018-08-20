# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:51:51 2018

@author: ANSHAY
"""
import numpy as np
import Activations

def initialize_parameters(layer_sizes, init_type='none'):
    """
    initializes Weight and Bias matrices for the neural network
    for given number of layer sizes
    inputs:
        layer_sizes : array of input_size and layer sizes such that
                  layer_sizes[0] contains input size
        inti_types: type of initialization.. he, Xavier,                   
    """
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
        parameters['B'+str(l)] = np.zeros((l,1))
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
    m = len(y_hat)
    regularization = np.sum(np.square([parameters['W'+str(i+1)] \
                                       for i in range(len(parameters)/2)]))
    regularization *= lamda/(2.0 * m)
    cost = -(np.dot(y, np.log(y_hat).T) + np.dot((1-y),\
                   np.log(1-y_hat).T) / m) + regularization
    return cost
  
def forward_propagate(data, Y, parameters, metadata):
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
    num_layers = len(parameters)/2
    for l in range(1, num_layers):
        # calculate Z = W*X + B
        Z_l = np.dot(parameters['W'+str(l)], data) + parameters['B'+str(l)]
        # calculate A = activtion function(Z)
        A_l = activation(Z_l, metadata['hidden_activation'])
        # put in caches
        caches['Z'+str(l)] = Z_l
        caches['A'+str(l)] = A_l
    # for final layer
        Z_L = np.dot(parameters['W'+str(num_layers)], data) + \
              parameters['B'+str(num_layers)]
        A_L = activation(Z_L, metadata['final_activation'])
        caches['Z'+str(num_layers)] = Z_L
        caches['A'+str(num_layers)] = A_L
    cost = compute_cost(Y, A_L, parameters, metadata['lamda'])

    return caches, cost

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


def backward_propagate(Y, parameters, caches, final_activation, hidden_activation):
    """
    computes gradients by backward propagation
    inputs:
        parameters- weights and biases for all layers , W and B
        caches containing activations and Z for every layer
    return:
        grads- gradients for weights and biases , dW, dB
    """
    num_layers = len(parameters)/2
    grads = {}
    
    A_L = caches['A'+str(num_layers)]
    Z_L = caches['Z'+str(num_layers)]
    dA_L = -np.divide((Y-A_L), np.dot(A_L, (1 - A_L)))
    dZ_L = activation_back(Z_L, dA_L, final_activation)
    
    dW_L = dZ_L * caches['A'+str(num_layers-1)]
    dB_L = dZ_L
    dA_L_1 = parameters['W'+str(num_layers)] * dZ_L
    grads['dW'+str(num_layers)] = dW_L
    grads['dB'+str(num_layers)] = dB_L
    grads['dA'+str(num_layers-1)] = dA_L_1
    for l in range(num_layers-1, 0):
        dA_l = grads['dA'+str(l)]
        dZ_l = activation_back(caches['Z'+str(l)], dA_l, hidden_activation)
        dW_l = dZ_l * caches['A'+str(l-1)]
        dB_l = dZ_l
        grads['dW'+str(l)] = dW_l
        grads['dB'+str(l)] = dB_l
        grads['dA'+str(l-1)] = parameters['W'+str(l)] * dZ_l
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
        parameters['W'+str(i)] -= learning_rate * grads['W'+str(i)]
        parameters['B'+str(i)] -= learning_rate * grads['B'+str(i)]
    
    return parameters

def train_model(train_data, train_Y, metadata):
    """
    trains and returns a neural network model for the given layer sizes,
    metadata, hyperparameters over the input data for the given number of
    iterations in metadata
    input:
        train_data: size -- (num_features x num_samples),
        train_Y: correct labels for training data. size- (1 x num_samples)
        metadata: contains-
            hidden_activation- type of activation function for hidden layers
            final_activation- type of activation function for final layer
            lamda- for cost computation
            iterations- number of iterations to train over
            init_type- type of initialization
            learning_rate- rate at which gradient descent moves/updates
    """
    layer_sizes = metadata['layer_sizes']
    iterations = metadata['iterations']
    parameters = initialize_parameters(layer_sizes, metadata['init_type'])
    for i in range(iterations):
        caches, cost = forward_propagate(train_data, train_Y, parameters, metadata)
        grads = backward_propagate(train_Y, parameters, caches,\
                                   metadata['final_activation'],\
                                   metadata['hidden_activation'])
        parameters = update_parameters(parameters, grads, metadata['learning_rate'])
        print ("Iteration " + str(i) + "Cost....................." + str(cost))
    
    model = {'parameters':parameters, 'caches':caches, 'cost':cost,\
             'metadata':metadata}
    
    return model