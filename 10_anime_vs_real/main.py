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
import config

## load data - change directories to the location of data
print("\nLoading data...")
train_dir = "D:\\Datasets\\anime_vs_real\\train\\"
val_dir = "D:\\Datasets\\anime_vs_real\\val\\"

train_data_gen = ImageDataGenerator(rescale=1./255,
                              rotation_range=40,
                              shear_range=0.2,
                              zoom_range=0.5,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True)
val_data_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_data_gen.flow_from_directory(
                    train_dir,
                    target_size=((config.Nrows, config.Ncols)),
                    batch_size=config.BATCH_SIZE,
                    class_mode='binary')
val_gen = val_data_gen.flow_from_directory(
                    val_dir,
                    target_size=((config.Nrows, config.Ncols)),
                    batch_size=config.BATCH_SIZE,
                    class_mode='binary')
print("\nData Generators defined")

## fetch model (training)
print("\nTraining model...")
model = model_gen.fetch_model(train_gen, val_gen)
print("\nTraining Complete")
