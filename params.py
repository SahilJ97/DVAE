HID_LAYER_SIZE = 200
BATCH_SIZE = 10
BATCH_SHAPE = (BATCH_SIZE, 28*28)
LATENT_SPACE_DIM = 50
EPOCH_GROUPS = [10, 10, 10, 10]
LEARNING_RATES = [.01, .005, .001, .0001]
TRAINING_EPOCHS = sum(EPOCH_GROUPS)
NOISE_STEP = .02
NOISE_BOUND = .501
LOG_FILE = 'log.csv'
BETA = 1000  # weighting coefficient for reconstruction error term
