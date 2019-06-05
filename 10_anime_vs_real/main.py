#!/usr/bin/python
# Filename: main.py
"""
Created on Thu May 30 00:23:55 2019

@author: Anshay
Main file for anime vs real classification

Runs anime vs real classification on selected dataset.
Loads data, visualizes data, fetch/train model, saves
model and evaluates the model.
"""

## import modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model_gen

## define metadata
print("\nDefining metadata")
metadata = {'Nrows':300, 'Ncols':300, 'BATCH_SIZE':64, 'NUM_EPOCHS':100,
            'FILTER_SIZE':(3,3)}
print("\nMetadata defined")

## load data - change directories to the location of data
print("\nLoading data...")
train_dir = "D:\\Datasets\\anime_vs_real\\train\\"
val_dir = "D:\\Datasets\\anime_vs_real\\val\\"

data_gen = ImageDataGenerator(rescale=1./255,
                              rotation_range=40,
                              shear_range=0.2,
                              zoom_range=0.5,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True)

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
