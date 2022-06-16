import tensorflow as tf
import numpy as np

#Loading the MNIST dataset
def loadDataset(mode='train', flatten_size=0):
    mnistDataset = tf.keras.datasets.mnist
    (x_train_val, y_train_val), (x_test, y_test) = mnistDataset.load_data()
    if mode == 'train':
        #Dividing the Dataset into training and validation set
        x_train, y_train =  x_train_val[:55000], y_train_val[:55000]
        x_validation, y_validation = x_train_val[55000:], y_train_val[55000:]

        #Reshaping the dataset
        x_train = x_train.reshape(x_train.shape[0], flatten_size)
        x_validation = x_validation.reshape(x_validation.shape[0], flatten_size)

        return x_train, y_train, x_validation, y_validation
    if mode == 'test':
        print(x_test.shape)
        x_test = x_test.reshape(x_test.shape[0], flatten_size)
        return x_test, y_test

#To randomize the dataset
def randomize(parameter, labels):
    permutations = np.random.permutation(labels.shape[0])
    shuffled_para = parameter[permutations, :]
    shuffled_labels = parameter[permutations]

    return shuffled_para, shuffled_labels