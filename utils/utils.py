import tensorflow as tf


def dist_l1(x):
    return tf.math.abs(x[0] - x[1])
