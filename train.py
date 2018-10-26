from params import *
import dvae
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from numpy.random import normal
from tensorflow import keras

_step = 0
_epoch = 0

'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
np.save('train_images', train_images)
np.save('test_images', test_images)
'''

train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')


def noiseless_batch():
    global _step, _epoch
    if BATCH_SIZE*(_step+1) > len(train_images):
        _epoch += 1
        _step = 0
    if _epoch < TRAINING_EPOCHS:
        _step += 1
        return np.reshape(train_images[(_step-1)*BATCH_SIZE:(_step)*BATCH_SIZE], BATCH_SHAPE)
    return None


def batch(level):
    noiseless = noiseless_batch()
    if noiseless is None:
        return None
    return inject_input_noise(noiseless, level), noiseless


def stochastic_noise(level):
    return uniform(-level, level, LATENT_SPACE_DIM)


def inject_input_noise(batch, level):
    noise = normal(0, level, BATCH_SHAPE)
    noisy = np.add(batch, noise)
    return noisy


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


x_noisy = tf.placeholder(tf.float64, BATCH_SHAPE, name='x_noisy')
x = tf.placeholder(tf.float64, BATCH_SHAPE, name='x')
means, standard_errors = dvae.encode(x_noisy)
variances = tf.square(standard_errors)

# generate a tensor from the standard normal distribution, and transform it in accordance with 'means' and 'variances'
r = tf.random_normal(shape=(BATCH_SIZE, LATENT_SPACE_DIM), dtype=tf.float64)
z = tf.add(tf.multiply(r, standard_errors), means)
ones = tf.constant(np.ones(LATENT_SPACE_DIM), dtype=tf.float64)
half = tf.constant(.5, dtype=tf.float64)

elbo = tf.cast(tf.losses.mean_squared_error(dvae.decode(z), x), tf.float64)
kl_divergence = tf.scalar_mul(half, tf.reduce_sum(
    tf.subtract(tf.subtract(tf.add(variances, means), tf.log(variances)), ones)))
kl_divergence = tf.minimum(kl_divergence, tf.constant(MAX_KL, dtype=tf.float64))
cost = tf.add(elbo, kl_divergence)

train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


if __name__ == '__main__':
    level = 1
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        while _epoch < TRAINING_EPOCHS:
            noisy_input, noiseless_input = batch(level)
            _means_, _se_, _elbo_, _kl_, _ = sess.run([means, standard_errors, elbo, kl_divergence, train_step],
                                       feed_dict={x_noisy: noisy_input, x: noiseless_input})
            print('EPOCH {} STEP {}\tELBO: {}, KL-Divergence: {}'.format(_epoch, _step, _elbo_, _kl_))
            if _step%30 == 0:
                print(_means_)
                print(_se_)

