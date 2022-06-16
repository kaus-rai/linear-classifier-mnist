import tensorflow as tf

#Loading the MNIST dataset
def loadDataset(mode='train'):
    mnistDataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnistDataset.load_data()
    if mode == 'train':
        return x_train, y_train
    if mode == 'test':
        return x_test, y_test

loadDataset()