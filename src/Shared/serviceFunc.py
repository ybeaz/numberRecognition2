import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

def imageprepare(argv):
    #"""
    #This function returns the pixel values.
    #The imput is a png file location.
    #"""
    im = Image.open(argv).convert('L')

    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    #print(tva)
    return np.array(tva).reshape((28,28))

def show_img(image_file_name, label):
  img = Image.open(image_file_name).convert('L').resize((28,28))
  img = ImageOps.invert(img)

  x, y = 10, 4
  plt.figure(figsize=(22,22))
  plt.subplot(y, x, 0+1).set_title('Label {label}'.format(label=label))
  plt.imshow(img.resize((28,28)),interpolation='nearest')
  plt.show()
    
def show_imgArr(imgArr, label):
  x, y = 10, 4
  plt.figure(figsize=(22,22))
  plt.subplot(y, x, 0+1).set_title('Predict {label}'.format(label=label))
  plt.imshow(imgArr.reshape((28,28)), interpolation='nearest')
  plt.show()