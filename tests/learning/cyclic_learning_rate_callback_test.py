import math

import tensorflow as tf
import numpy as np

from tavolo.learning import CyclicLearningRateCallback


def test_logic():
    """ Test logic on known input """

    # Input
    input_2d = np.random.normal(size=(1000, 20))
    labels = np.random.randint(low=0, high=2, size=1000)

    # Create model
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(20,)),
                                 tf.keras.layers.Dense(10, activation=tf.nn.relu),
                                 tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='binary_crossentropy')

    # Create callback
    clr = CyclicLearningRateCallback()
    expected_lr_values = list(np.linspace(0.001, 0.006, 2000))
    expected_lr_values = expected_lr_values + expected_lr_values[::-1]  # Triangle

    # Run model
    model.fit(input_2d, labels, batch_size=10, epochs=5, callbacks=[clr], verbose=0)

    assert all(math.isclose(a, b, rel_tol=0.001) for a, b in zip(clr.history['lr'], expected_lr_values))