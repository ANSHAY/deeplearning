# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:48:17 2019

@author: Anshay
Predicts anime or real on user uploaded image from a saved model
"""

import tensorflow as tf
import input_image
from tkinter import filedialog

## predict for new uploaded images
def predict(model):
    img = input_image.inpImg()
    print("\nPredicting...")
    prediction = model.predict(img)
    print(prediction)
    print("\nThe image belongs to a-- ")
    if prediction > 0.5:
        print ("Real")
    else:
        print ("Anime")

if __name__=="__main__":
    model_path = filedialog.askopenfilename()
    ## load model
    print("\nLoading Model...")
    model = tf.keras.models.load_model(model_path)
    print("\nPredicting on input image...")
    again = 'y'
    while (again == 'y' or again == 'Y'):
        predict(model)
        again = input("\nPredict more?(y/n):   ")
