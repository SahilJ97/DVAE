import numpy as np
from numpy.random import normal
from params import *


if __name__ == '__main__':

    train = np.empty((0, 28, 28))
    # generate from this.
    for i in range(TRAIN_SET_SIZE):
        latent = normal(.5, .1, (LATENT_SPACE_DIM))
        image = np.zeros(28*28)
        for j in range(LATENT_SPACE_DIM):
            for k in range(10*j, 10*j + 10):
                image[k] += latent[j]  # no, need to make it a function!
        train = np.append(train, [np.reshape(image, (28, 28))], axis=0)
    print(np.shape(train))
    np.save('toy_train', train)

    test = np.empty((0, 28, 28))
    # generate from this.
    for i in range(TEST_SET_SIZE):
        latent = normal(.5, .1, (LATENT_SPACE_DIM))
        image = np.zeros(28*28)
        for j in range(LATENT_SPACE_DIM):
            for k in range(10 * j, 10 * j + 10):
                image[k] += latent[j]
        test = np.append(test, [np.reshape(image, (28, 28))], axis=0)
    print(np.shape(test))
    np.save('toy_test', test)
