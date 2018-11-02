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
beta = tf.constant(BETA, dtype=tf.float64)

'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
np.save('train_images', train_images)
np.save('test_images', test_images)
'''


def inject_noise(batch, level):
    shape = np.shape(batch)
    noise = normal(.5, level, shape)
    noisy = np.add(batch, noise)
    noisy = np.maximum(np.minimum(noisy, np.ones(shape)), np.zeros(shape))  # constrain pixels to [0, 1]
    return noisy


train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')
f = plt.figure()
plt.imsave('image.png', test_images[2], cmap=plt.get_cmap('gray'))
plt.imsave('max_noise_image.png', inject_noise(test_images[2], NOISE_BOUND), cmap=plt.get_cmap('gray'))
plt.close(f)
test_images = np.reshape(test_images, (len(test_images), -1))
test_shape = np.shape(test_images)


def noiseless_batch():
    global _step, _epoch
    if BATCH_SIZE*(_step+1) > len(train_images):
        print('Epoch {} complete'.format(_epoch))
        _epoch += 1
        _step = 0
    if _epoch < TRAINING_EPOCHS:
        _step += 1
        return np.reshape(train_images[(_step-1)*BATCH_SIZE:(_step)*BATCH_SIZE], BATCH_SHAPE)
    return None


def get_learning_rate():
    for i in range(len(EPOCH_GROUPS)):
        if _epoch < sum(EPOCH_GROUPS[:i+1]):
            return LEARNING_RATES[i]
    return None


def batch(level):
    noiseless = noiseless_batch()
    if noiseless is None:
        return None
    return inject_noise(noiseless, level), noiseless


def stochastic_noise(level):
    return uniform(-level, level, LATENT_SPACE_DIM)


x_noisy = tf.placeholder(tf.float64, BATCH_SHAPE, name='x_noisy')
x = tf.placeholder(tf.float64, BATCH_SHAPE, name='x')
means, standard_errors = dvae.encode(x_noisy)
variances = tf.square(standard_errors)

# generate a tensor from the standard normal distribution, and transform it in accordance with 'means' and 'variances'
r = tf.placeholder(tf.float64, (BATCH_SHAPE[0], LATENT_SPACE_DIM))
z = tf.add(tf.multiply(r, standard_errors), means)

ones = tf.constant(np.ones(LATENT_SPACE_DIM), dtype=tf.float64)
half = tf.constant(.5, dtype=tf.float64)
reconst_error = tf.cast(tf.losses.mean_squared_error(dvae.decode(z), x), tf.float64)  # THIS NEEDS FIXING!!! LEARNS TO GENERATE ONE IMAGE!
kl_divergence = tf.scalar_mul(half, tf.reduce_sum(
    tf.subtract(tf.subtract(tf.add(variances, means), tf.log(variances)), ones)))
cost = tf.add(tf.multiply(reconst_error, beta), kl_divergence)
learning_rate = tf.placeholder(dtype=tf.float64, shape=(), name='learning_rate')
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

test_noiseless = tf.constant(test_images, dtype=tf.float64)
test_noisy = tf.placeholder(tf.float64, test_shape)

test_means, test_standard_errors = dvae.encode(test_noisy)
test_variances = tf.square(test_standard_errors)

t_r = tf.placeholder(tf.float64, (test_shape[0], LATENT_SPACE_DIM))
t_z = tf.add(tf.multiply(t_r, test_standard_errors), test_means)

t_reconst_error = tf.cast(tf.losses.mean_squared_error(dvae.decode(t_z), test_noiseless), tf.float64)
t_kl_divergence = tf.scalar_mul(half, tf.reduce_sum(
    tf.subtract(tf.subtract(tf.add(test_variances, test_means), tf.log(test_variances)), ones)))

#demo_image = dvae.decode(tf.slice(t_z, (2, 0), (1, LATENT_SPACE_DIM)))
demo_image = dvae.decode(tf.constant(np.array([np.zeros(LATENT_SPACE_DIM)]), dtype=tf.float64))


def save_demo_image(fname, pixels):
    fig = plt.figure()
    plt.imsave('{}.png'.format(fname), np.reshape(pixels, (28, 28)), cmap=plt.get_cmap('gray'))
    plt.close(fig)


def write_test_performance(ep, it, sess, train_nlevel):
    for test_nlevel in np.arange(0., NOISE_BOUND, NOISE_STEP):
        _rec_er_, _kl_, _demo_ = sess.run([t_reconst_error, t_kl_divergence, demo_image],
                                  feed_dict={test_noisy: inject_noise(test_images, test_nlevel),
                                             t_r: normal(0., 1., (test_shape[0], LATENT_SPACE_DIM))})
        with open(LOG_FILE, 'a') as file:
            file.write('{} {}, {}, {}, {}, {}, {}\n'
                       .format(train_nlevel, it, test_nlevel, ep, _rec_er_, _kl_, BETA*_rec_er_ + _kl_))
        save_demo_image('{}{}{}'.format(ep, it, test_nlevel, train_nlevel), _demo_)


def train(train_nlevel, it):
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        while _epoch < TRAINING_EPOCHS:
            b = batch(train_nlevel)
            if b is None:
                break
            noisy_input, noiseless_input = b
            c, _ = sess.run([cost, train_step],
                               feed_dict={x_noisy: noisy_input, x: noiseless_input, learning_rate: get_learning_rate(),
                                          r: normal(0., 1., (BATCH_SHAPE[0], LATENT_SPACE_DIM))})
            if c == np.inf or c == np.nan:
                print('Representation error. Breaking loop.')
                return
            if _step == 1 and (_epoch%10 == 1 or _epoch == 0):
                write_test_performance(_epoch, it, sess, train_nlevel)
        write_test_performance('final', it, sess, train_nlevel)


if __name__ == '__main__':
    with open(LOG_FILE, 'w') as file:
        file.write('training noise-level and iteration, test noise-level, epoch, reconstruction error, kl-divergence, '
                   'cost\n')
    for noise_level in np.arange(0., NOISE_BOUND, NOISE_STEP):
        print('Noise level: {}'.format(noise_level))
        for i in range(3):
            print('Iteration: {}'.format(i))
            _epoch, _step = 0, 0
            train(noise_level, i)
