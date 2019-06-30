# --- Import libraries --- #
import warnings
warnings.filterwarnings("ignore")

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as loadmata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# --- Import project files --- #
from src.Shared.serviceFunc import *
from src.Teaching.mnist_cnn import *

# --- import MNIST from local files --- #
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

with open('C:/Data/Dev/NumberRecognition/assets/train-images-idx3-ubyte.gz', 'rb') as f:
  x_train = extract_images(f)
with open('C:/Data/Dev/NumberRecognition/assets/train-labels-idx1-ubyte.gz', 'rb') as f:
  y_train = extract_labels(f)
with open('C:/Data/Dev/NumberRecognition/assets/t10k-images-idx3-ubyte.gz', 'rb') as f:
  x_test = extract_images(f)
with open('C:/Data/Dev/NumberRecognition/assets/t10k-labels-idx1-ubyte.gz', 'rb') as f:
  y_test = extract_labels(f)

print ('')
print ("Number of images for training:", x_train.shape[0])
print ("Number of images used for testing:", x_test.shape[0])
pix = int(np.sqrt(x_train.shape[1]))
print ("Each image is:", pix, "by", pix, "pixels")

# --- Show first image --- #
""" plt.figure(figsize=(22,22))
x, y = 10, 4

for i in range(4):
    
    label = y_train[i]
    plt.subplot(y, x, i+1).set_title('Digit {label}'.format(label=label))
    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
    
plt.show() """

# --- mnist_cnn.py --- #
modelOutcome = teachMnistCnn()

# ---  --- #
image_file_name = 'C:/Data/Dev/NumberRecognition/samples/9_3.png'
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

predict = modelOutcome.predict(xTestReshape)
predictClasses = modelOutcome.predict_classes(xTestReshape)
print('[10 predict_classes]', predictClasses, 'predict', predict)
show_imgArr(imgArr, predictClasses[0])

# ---  --- #


# ---  --- #


# ---  --- #