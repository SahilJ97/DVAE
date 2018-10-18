import tensorflow as tf
from params import *


def encode(x):
    # x:        input data, or observation
    # returns:  mean and variance vectors for the learned posterior distribution

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        hidden_layer_1 = tf.math.softplus(tf.layers.dense(inputs=x, units=HID_LAYER_SIZE))
        output_layer = tf.math.softplus(tf.layers.dense(inputs=hidden_layer_1, units=2*LATENT_SPACE_DIM))
        return output_layer


def decode(z):
    # z:        latent variable
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        hidden_layer_1 = tf.math.softplus(tf.layers.dense(inputs=z, units=HID_LAYER_SIZE))
        hidden_layer_2 = tf.sigmoid(tf.layers.dense(inputs=hidden_layer_1, units=HID_LAYER_SIZE))
        return hidden_layer_2
