# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:34:05 2018

This loads the data, defines hyperparameter, calls the neural network on train
data, tests the model on test data, prints accuracy, plots graphs

@author: ANSHAY
"""
import numpy as np
import pickle
import neural_network as NN
import load_data as load
import matplotlib.pyplot as plt
from PIL import Image

def load_data(dataset):
    """
    loads the data for learning and separates into training and test data
    """
    if dataset == 'cat':
        train_X, train_Y, test_X, test_Y, classes = load.loadCatData('E:\Datasets\catvsnoncat')
    elif dataset == 'planar':
        train_X, train_Y, test_X, test_Y, classes = load.loadPlanarData()
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

def pre_process(data):
    """
    pre process the data reshaping it into (num_features x num_samples)
    """
    return data.reshape(data.shape[0], -1).T

def load_hyperparameters(filename):
    """
    loads metadata from external file named metadata
    metadata: contains-
        hidden_activation- type of activation function for hidden layers
        final_activation- type of activation function for final layer
        lamda- regularisation parameter for cost computation
        iterations- number of iterations to train over
        init_type- type of initialization
        learning_rate- rate at which gradient descent moves/updates
        layer_sizes- array of size of each layer of neural network
    """
    try:
        f = open(filename, 'r')
        metadata = pickle.load(f)
        f.close()
    except:
        metadata = {'hidden_activation': 'relu', 'final_activation': 'sigmoid',\
            'lamda': 0.0001, 'iterations': 300, 'init_type': 'xavier',\
            'learning_rate': 0.1, 'layer_sizes': [20, 7, 5, 1]}
        f = open(filename, 'wb')
        pickle.dump(metadata, f)
        f.close()
    return metadata

def train_model(train_X, train_Y, metadata, dataset):
    """
    loads the saved model if exist else make new and save
    """
    try:
        f = open('models/nn_model_'+ metadata['hidden_activation'] + '_' +\
                 str(metadata['iterations']) + '.pkl', 'rb')
        model = pickle.load(f)
        f.close()
        print ('\nModel found................')
    except:
        print ('\nModel not found............')
        model = NN.train_model(train_X, train_Y, metadata)
        f = open('models/nn_model_'+ metadata['hidden_activation'] + '_' +\
                 str(metadata['iterations']) + '.pkl', 'wb')
        pickle.dump(model, f)
        f.close()
    return model

def plot_all():
    return

def build_and_test_model():
    """
    Main program that defines parameters of the neural network and calls the
    functions for training testing and plotting the data
    """
    ## load data
    dataset = 'cat'
    train_X, train_Y, test_X, test_Y, classes = load_data(dataset)
    num_px_x = train_X.shape[1]
    num_px_y = train_X.shape[2]
    
    ## standardize data
    train_X, train_mean, train_deviation = standardize(train_X)
    test_X, _, _ = standardize(test_X, train_mean, train_deviation)
    
    ## pre process the data
    train_X = pre_process(train_X)
    test_X = pre_process(test_X)
    
    ## define metadata
    metadata = load_hyperparameters('metadata.pkl')
    metadata['classes'] = classes
    metadata['train_mean'] = train_mean
    metadata['train_deviation'] = train_deviation
    metadata['input_size'] = (num_px_x, num_px_y)
    
    ## define model
    model = train_model(train_X, train_Y, metadata, dataset)
    
    ## test model
    accuracy = NN.test_model(model, train_X, train_Y)
    print ('accuracy for training data..................' + str(accuracy))
    accuracy = NN.test_model(model, test_X, test_Y)
    print ('accuracy for test data......................' + str(accuracy))

    return model

def test_image(model, image_path):
    classes = model['metadata']['classes']
    print (classes)
    img = Image.open(image_path)
    plt.imshow(img)
    new_shape = model['metadata']['input_size']
    img = img.resize(new_shape, Image.ANTIALIAS)
    img, _, _ = standardize(img, model['metadata']['train_mean'], model['metadata']['train_deviation'])
    img = np.array(img).reshape((new_shape[0]*new_shape[1]*3, 1))
    prediction = np.squeeze(NN.predict(model, img))
    if prediction:
        print ('The given model predicts this as a CAT image')
    else:
        print ('The given model predicts this as a NON-CAT image')
        

if __name__ == '__main__':
    build_and_test_model()
    f = open('models/nn_model_relu_300.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    image_path = 'images/cat (6).jpg'
    test_image(model, image_path)
    
    
    
