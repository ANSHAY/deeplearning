# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:05:57 2018

@author: anshay
"""

import numpy as np
import Utils as utils

def zero_pad(X, pad):
    """
    adds zero padding to the input matrix by pad number of padding layers
    X: m x height x width x #channels
    pad: scalar
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad

def conv_single(a_window, w, b):
    """
    performs single step of the convolution
    a_window: a window on which filter is applied. size: (f, f, n_C)
    w: weights of the filter, size: (f, f, n_C)
    b: bias value for the filter, scalar
    """
    return np.sum(a_window * w) + b

def forward_propagation(a_prev, W, b, h_param):
    '''
    a_prev: output A of previous layer, size: (m, n_Hprev, n_Wprev, n_Cprev)
    W:      weights of current layer, size: (f, f, n_Cprev, n_Cnew)
    b:      bias values for current layer, size: (1, 1, 1, n_Cnew)
    '''
    m, n_Hprev, n_Wprev, _ = a_prev.shape
    f = W.shape[0]
    stride = h_param['stride']
    pad = h_param['pad']
    # calculate size of output
    n_Hnew = int(1 + (n_Hprev + 2 * pad - f) / stride)
    n_Wnew = int(1 + (n_Wprev + 2 * pad - f) / stride)
    n_Cnew = W.shape[-1]
    Z = np.zeros((m, n_Hnew, n_Wnew, n_Cnew))
    for i in range(m):
        a_prev_i = a_prev[i]
        for h in range(n_Hprev):
            for w in range(n_Wprev):
                # find corners of window slice
                vert_start = h * stride
                vert_stop = vert_start + f
                hor_start = w * stride
                hor_stop = hor_start + f
                for c in range(n_Cnew):
                    Z[i, h, w, c] = conv_single(a_prev_i[vert_start:vert_stop,\
                                                hor_start:hor_stop, :],\
                                                W[:, :, :, c], b[:, :, :, c])
    return Z

def pool(a_prev, h_param, pool_type='max'):
    '''
    performs pooling on the activations of current layer
    a_prev: output A of previous layer, size: (m, n_Hprev, n_Wprev, n_Cprev)
    pool_type: type of pooling-> max or average
    '''
    m, n_Hprev, n_Wprev, n_Cprev = a_prev.shape
    stride = h_param['stride']
    f = h_param['f']
    
    # calculate size of output
    n_Hnew = int(1 + (n_Hprev - f) / stride)
    n_Wnew = int(1 + (n_Wprev - f) / stride)
    n_Cnew = n_Cprev
    
    a_pool = np.zeros((m, n_Hnew, n_Wnew, n_Cnew))
    
    for i in range(m):
        a_prev_i = a_prev[i]
        for h in range(n_Hprev):
            for w in range(n_Wprev):
                # find corners of window slice
                vert_start = h * stride
                vert_stop = vert_start + f
                hor_start = w * stride
                hor_stop = hor_start + f
                for c in range(n_Cnew):
                    a_slice = a_prev_i[vert_start:vert_stop, hor_start:hor_stop]
                    if pool_type == 'max':
                        a_pool[i, h, w, c] = np.max(a_slice)
                    elif pool_type == 'average':
                        a_pool[i, h, w, c] = np.average(a_slice)
    return a_pool

def forward_activation(Z, activation):
    '''
    returns the asked activation for current layer
    '''
    if activation == 'softmax':
        A = utils.softmax(Z)
    elif activation == 'relu':
        A = utils.relu(Z)
    return A

def forward(X, W, B, h_param, num_layers):
    """
    forward propagation and activation for every layer
    X: input data: (m, n_H, n_W, n_C)
    W: filters: (f, f, n_Cprev, n_Cnew, numLayers)
    num_layers: # of layers in the network
    """
    cache = {'A0' : X}
    for l in range(1, num_layers):
        Z = forward_propagation(cache['A' + str(l)], W, B, h_param)
        A = forward_activation(Z, 'relu')
        A_pool = pool(A, h_param, 'max')
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A_pool
    return cache

def backward():
    """
    returns gradients calculated by backward propagation
    """
    return grads

    
    
