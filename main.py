import tensorflow as tf
from datasetHelper import loadDataset, getNextBatch
from modelHelper import weight_variable, bias_variable
from sklearn.utils import shuffle

tf.compat.v1.disable_eager_execution()
#defining the dimensions of the dataset images
#MNIST dataset are 28X28
img_h, img_w = 28, 28

#defining total number of pixels
img_flatten = img_h*img_w

#Defining the number of classes
classes = 10
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.001

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

#Creating the tensor Placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_flatten], name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, classes], name='Y')

#Weight initialized randomly
W = weight_variable(shape=[img_flatten, classes])
b = bias_variable(shape=[classes])

logits = tf.linalg.matmul(x, W) + b

loss = tf.math.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits), name="loss"
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name="Adam-op").minimize(loss)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1), name="correct_pred")
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

var_init = tf.compat.v1.global_variables_initializer()

#Creating the interactive session to train the model
sess = tf.compat.v1.InteractiveSession()
sess.run(var_init)

iter_size = int(len(y_train)/batch_size)

for epoch in range(epochs):
    print("Training Epochs : ", (epoch+1))
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    for iteration in range(iter_size):
        start = iteration*batch_size
        end = (iteration+1)*batch_size
        X_batch, y_batch = getNextBatch(X_train, y_train, start, end)

        dict_batch = {
            x : X_batch,
            y : y_batch
        }

        sess.run(optimizer, feed_dict=dict_batch)

        if(iteration%display_freq == 0):
            loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=dict_batch)
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))

    #Running Validation on every epoch
    feed_dict_valid = {
        x : X_validation[:1000],
        y : y_validation[:1000],
    }

    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')


#Accuracy on Test Set
print('---------------------------------------------------------')
print('Checking the accuracy on Test Set')
feed_dict_test = {
        x : X_test[:1000],
        y : y_test[:1000],
    }
loss_test, acc_test = sess.run([loss,accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')




