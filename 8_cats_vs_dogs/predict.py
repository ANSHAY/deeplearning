# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:48:17 2019

@author: Anshay
Predicts cat or dog on user uploaded image from a saved model
"""
import tensorflow as tf
import input_image
from tkinter import filedialog
import numpy as np
from PIL import Image

## predict for new uploaded images
def predict(model_path=None, test_list=None):
    ## load model
    print("\nLoading Model...")
    if (not model_path):
        model_path = filedialog.askopenfilename()
    model = tf.keras.models.load_model(model_path)
    print("\nPredicting on input image...")
    inpImg = input_image.inpImg()
    print("\nPredicting...")
    prediction = model.predict(inpImg)
    print(prediction)
    print("\nThe image belongs to a-- ")
    if prediction > 0.5:
        print ("DOG")
    else:
        print ("CAT")

if __name__=="__main__":
    predict()
