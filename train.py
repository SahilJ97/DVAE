from params import *
import dvae
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def cost(x, z):



if __name__ == '__main__':
    with tf.Session() as sess:

