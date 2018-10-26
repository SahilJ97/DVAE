import tensorflow as tf
from params import *


def encode(x):
    # x:        input data, or observation
    # returns:  mean and standard error vectors for the learned posterior distribution
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        hidden_layer_1 = tf.nn.softplus(tf.layers.dense(inputs=x, units=HID_LAYER_SIZE))
        means = tf.nn.softplus(tf.layers.dense(inputs=hidden_layer_1, units=LATENT_SPACE_DIM))
        standard_errors = tf.nn.softplus(tf.layers.dense(inputs=hidden_layer_1, units=LATENT_SPACE_DIM))
        return means, standard_errors


def decode(z):
    # z:        latent variable
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        hidden_layer_1 = tf.nn.softplus(tf.layers.dense(inputs=z, units=HID_LAYER_SIZE))
        hidden_layer_2 = tf.sigmoid(tf.layers.dense(inputs=hidden_layer_1, units=BATCH_SHAPE[1]))
        return hidden_layer_2
