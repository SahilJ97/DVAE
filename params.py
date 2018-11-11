HID_LAYER_SIZE = 200
BATCH_SIZE = 10
BATCH_SHAPE = (BATCH_SIZE, 28*28)
LATENT_SPACE_DIM = 50
EPOCH_GROUPS = [10, 10, 20, 20]
LEARNING_RATES = [.01, .005, .001, .0001]
TRAINING_EPOCHS = sum(EPOCH_GROUPS)
NOISE_STEP = .02
MAX_N_LEVEL = .501
ALPHA = .1  # standard error of the distribution from which r's entries are drawn--should be contained in [0, 1]
BETA = 1.  # weighting coefficient for reconstruction error term.
WRITE_EPOCHS = [0, 1, 2, 3, 6, 11, 16, 21, 26, 31, 41, 51]
IMAGE_DIR = 'Images/'
TRAIN_SET_SIZE = 60000
TEST_SET_SIZE = 10000
TOY = True

if TOY is True:
    LOG_FILE = 'toy_log.csv'
    TRAIN_IMAGES = 'toy_train.npy'
    TEST_IMAGES = 'toy_test.npy'
else:
    LOG_FILE = 'log.csv'
    TRAIN_IMAGES = 'train_images.npy'
    TEST_IMAGES = 'test_images.npy'
