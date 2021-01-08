
# coding: utf-8

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
from matplotlib.image import imread
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import PIL
from PIL import Image, ImageFile
import cv2
import os
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
import seaborn as sns
ImageFile.LOAD_TRUNCATED_IMAGES = True
########## SETTINGS

pd.set_option('display.max_columns', None)
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

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


def prepare_image(path, image_size=256):
    # import
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize
    image = cv2.resize(image, (int(image_size), int(image_size)))

    # convert to tensor
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

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


if __name__ = "__main__":
    app.run()

