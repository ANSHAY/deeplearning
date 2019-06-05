#!/usr/bin/python
# Filename: main.py
"""
Created on Thu May 30 00:23:55 2019

@author: Anshay
Main file for Apron defected vs healthy classification

Runs defected vs healthy classification on selected dataset.
Loads dataa, visualizes data, fetch/train model, saves
model and evaluates the model.
"""

## import modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model_gen

## define metadata
print("\nDefining metadata")
metadata = {'Nrows':300, 'Ncols':300, 'BATCH_SIZE':20, 'NUM_EPOCHS':70,
            'FILTER_SIZE':(5,5)}
print("\nMetadata defined")

## load data - change directories to the location of data
print("\nLoading data...")
train_dir = "D:\\Datasets\\Fender_apron\\train\\"
val_dir = "D:\\Datasets\\Fender_apron\\val\\"

data_gen = ImageDataGenerator(rescale=1./255,
                              rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              zoom_range=0.5,
                              horizontal_flip=True,
                              vertical_flip=True)

train_gen = data_gen.flow_from_directory(
                    train_dir,
                    target_size=((metadata['Nrows'], metadata['Ncols'])),
                    batch_size=metadata['BATCH_SIZE'],
                    class_mode='binary')

val_gen = data_gen.flow_from_directory(
                    val_dir,
                    target_size=((metadata['Nrows'], metadata['Ncols'])),
                    batch_size=metadata['BATCH_SIZE'],
                    class_mode='binary')

print("\nData Generators defined")

## visualize data

## preprocess data

## fetch model (training)
print("\nTraining model...")
model = model_gen.fetch_model(train_gen, val_gen, metadata)
print("\nTraining Complete")
