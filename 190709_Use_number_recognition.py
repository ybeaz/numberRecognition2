# --- Import libraries --- #
import warnings
warnings.filterwarnings("ignore")

#import sys
#import os

import tensorflow as tf

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import scipy as loadmata

from keras.models import load_model
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# --- Import project files --- #
from src.Shared.serviceFunc import *

loadedModel = load_model('C:/Data/Dev/NumberRecognition2/models/weightsfile.h5')

# ---  --- #
image_file_name = 'C:/Data/Dev/NumberRecognition2/samples/7_3.png'
#show_img(image_file_name, 2)

img = Image.open(image_file_name).convert('L').resize((28,28))
#img = change_contrast(img, 0)
img = ImageOps.invert(img)

imgArr = imageprepare(image_file_name)
imgArr = np.array(imgArr)

imgArrNp = np.array(imgArr)
#xTestReshape = imgArrNp.reshape(1,28,28,1) 
xTestReshape = imgArrNp[np.newaxis, ..., np.newaxis]
#print('[8]',xTestReshape.shape)
#print('[9]',imgArr)

predict = loadedModel.predict(xTestReshape)
predictClasses = loadedModel.predict_classes(xTestReshape)
print('[10 predict_classes]', predictClasses, 'predict', predict)
show_imgArr(imgArr, predictClasses[0])

# ---  --- #


# ---  --- #


# ---  --- #