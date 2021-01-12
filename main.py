
# coding: utf-8


import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
import PIL
from PIL import Image, ImageFile
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pickle
import random
import time
import sys
import os
import math


app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():

    inputQuery1 = request.form['img']

    p = prepare_image(inputQuery1)
    image233 = train_trans(p)
    image233 = image233.unsqueeze(0)

    image233 = image233.to(device)
    model = pickle.load(open("model2.sav", "rb"))
    outputs10 = model(image233)
    _, pred = torch.max(outputs10.data, 1)


    if pred == 0:
        output = "No DR"
    elif pred == 1:
        output = "Mild"
    elif pred == 2:
        output = "Moderate"
    elif pred == 3:
        output = "Severe"
    elif pred == 4:
        output = "Proliferative DR"

    return render_template('home.html', output=output, img = request.form['img'])


def prepare_image(path, sigmaX = 10, do_random_crop = False):
    
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


##### automatic crop of black areas
def crop_black(img, tol = 7):
    
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img  = np.stack([img1, img2, img3], axis = -1)
            return img
        
        
##### circular crop around image center
def circle_crop(img, sigmaX = 10):   
        
    height, width, depth = img.shape
    
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness = -1)
    
    img = cv2.bitwise_and(img, img, mask = circle_img)
    return img 


##### random crop
def random_crop(img, size = (0.9, 1)):

    height, width, depth = img.shape
    
    cut = 1 - random.uniform(size[0], size[1])
    
    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]    
    
    return img

train_trans = torchvision.transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomRotation((-360, 360)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()
                                 ])


p = prepare_image(img_name122)
image233 = train_trans(p)
image233 = image234.unsqueeze(0)

image233 = image234.to(device)
outputs10 = load_model2(image234)
_,pred = torch.max(outputs10.data,1)
print(pred)


if __name__ == "__main__":
    app.run()

