# -*- coding: utf-8 -*-
"""
Created on Thu May 30 04:33:24 2019

@author: Anshay
Input Image from user for testing model
"""

import tkinter as tk
from tkinter import filedialog

import numpy as np
from tensorflow.keras.preprocessing import image
import config

root = tk.Tk()
root.withdraw()
def inpImg():
    filePath = filedialog.askopenfilename()
    img = image.load_img(filePath, target_size=(config.Nrows, config.Ncols))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    return x
