import tensorflow as tf

#Weight Distribution Definition
def weight_variable(shape):
    init = tf.compat.v1.truncated_normal_initializer(stddev=0.01)
    return tf.compat.v1.get_variable(
        'W',
        dtype=tf.float32,
        shape=shape,
        initializer=init
    )

#Bias Distribution Definition
def bias_variable(shape):
    init = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.compat.v1.get_variable(
        'b',
        dtype=tf.float32,
        initializer=init
    )