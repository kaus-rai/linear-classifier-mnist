import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from datasetHelper import loadDataset


#defining the dimensions of the dataset images
#MNIST dataset are 28X28
img_h, img_w = 28, 28

#defining total number of pixels
img_flatten = img_h*img_w

#Defining the number of classes
classes = 10

#Load MNIST dataset
X_train, y_train, X_validation, y_validation = loadDataset(mode='train', flatten_size=img_flatten)
X_test, y_test = loadDataset(mode='test', flatten_size=img_flatten)
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_validation)))

#Checking the shape of the train and validation set
print('x_train:\t{}'.format(X_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_validation:\t{}'.format(X_validation.shape))
print('y_validation:\t{}'.format(y_validation.shape))



