import tensorflow as tf
import numpy as np

from tavolo.learning import LearningRateFinder


def test_logic():
    """ Test logic on known input """

    # Input
    input_2d = np.random.normal(size=(10000, 20))
    labels = np.random.randint(low=0, high=2, size=10000)

    # Create model
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(20,)),
                                 tf.keras.layers.Dense(10, activation=tf.nn.relu),
                                 tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='binary_crossentropy')

    # Learning rate range test
    lr_finder = LearningRateFinder(model=model)

    # Run model
    lrs, losses = lr_finder.scan(input_2d, labels, batch_size=50)

    assert len(lrs) == len(losses)
