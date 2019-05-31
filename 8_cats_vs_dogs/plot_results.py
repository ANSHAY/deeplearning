# -*- coding: utf-8 -*-
"""
Created on Thu May 30 04:17:32 2019

@author: Anshay
Plots loss and Accuracy of a model
"""

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

def plot_results(history):
    # Retrieve a list of list results on training and val data
    # sets for each training epoch
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    # Get number of epochs
    epochs=range(len(acc)) 
    
    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()
    
    # Plot training and validation loss per epoch
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    
    plt.title('Training and validation loss')
