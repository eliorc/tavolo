from typing import Optional, Callable, Any

import numpy as np
import pytest
import tensorflow as tf
from _pytest.fixtures import FixtureRequest

import tavolo as tvl


# <editor-fold desc="Fixtures">
@pytest.fixture(scope='function')
def binary_classification_dataset() -> tf.data.Dataset:
    """
    TensorFlow dataset instance with binary labels

    :return: Binary classification dataset
    """

    # Create features and labels
    X = tf.random.normal(shape=(100, 3))
    y = tf.random.uniform(minval=0, maxval=2, dtype=tf.int32, shape=(100,))  # Binary labels

    return tf.data.Dataset.from_tensor_slices((X, y))


@pytest.fixture(scope='session')
def multi_class5_classification_dataset() -> tf.data.Dataset:
    """
    TensorFlow dataset instance with multi-class labels as one hot (5 classes)

    :return: Multi-class one-hot classification dataset
    """

    # Create features
    X = tf.random.normal(shape=(100, 3))

    # Create one multi-class (one hot) labels
    y = tf.random.normal(shape=(100, 5))
    y = tf.one_hot(tf.argmax(y, axis=-1), depth=5)

    return tf.data.Dataset.from_tensor_slices((X, y))


@pytest.fixture(scope='session')
def multi_class5_classification_dataset_sparse_labels() -> tf.data.Dataset:
    """
    TensorFlow dataset instance with multi-class sparse labels (5 classes)

    :return: Multi-class sparse (labels) classification dataset
    """

    # Create features
    X = tf.random.normal(shape=(100, 3))

    # Create one multi-class (one hot) labels
    y = tf.random.uniform(minval=0, maxval=5, dtype=tf.int32, shape=(100,))

    return tf.data.Dataset.from_tensor_slices((X, y))


@pytest.fixture(scope='session')
def multi_label5_classification_dataset() -> tf.data.Dataset:
    """
    TensorFlow dataset instance with multi-label sparse labels (5 classes)

    :return: Multi-label sparse (labels) classification dataset
    """

    # Create features
    X = tf.random.normal(shape=(100, 3))

    # Create one multi-class (one hot) labels
    y = tf.random.normal(shape=(100, 5))
    y = tf.cast(y > 0.5, dtype=tf.int32)

    return tf.data.Dataset.from_tensor_slices((X, y))


@pytest.fixture(scope='function')
def binary_classification_model() -> tf.keras.Model:
    """
    Binary classification model, already compiled

    :return: Model
    """

    # Build model
    model = tf.keras.Sequential(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy())

    return model


@pytest.fixture(scope='function')
def multi_class5_classification_model() -> tf.keras.Model:
    """
    Multi class classification model (5 classes), already compiled

    :return: Model
    """

    # Build model
    model = tf.keras.Sequential(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy())

    return model


@pytest.fixture(scope='function')
def multi_class5_classification_model_sparse_labels() -> tf.keras.Model:
    """
    Multi class classification model (5 classes), already compiled for sparse labels

    :return: Model
    """

    # Build model
    model = tf.keras.Sequential(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy())

    return model


@pytest.fixture(scope='function')
def multi_class5_classification_model_logits() -> tf.keras.Model:
    """
    Multi class classification model (5 classes), already compiled and outputs logits (not probabilities)

    :return: Model
    """

    # Build model
    model = tf.keras.Sequential(tf.keras.layers.Dense(5, activation=None))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    return model


@pytest.fixture(scope='function')
def multi_class5_classification_model_logits_sparse_labels() -> tf.keras.Model:
    """
    Multi class classification model (5 classes), already compiled, outputs logits (not probabilities) and sparse labels

    :return: Model
    """

    # Build model
    model = tf.keras.Sequential(tf.keras.layers.Dense(5, activation=None))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    return model


# </editor-fold>


@pytest.mark.parametrize(
    argnames=['dataset', 'model', 'outputs_to_probabilities', 'sparse_labels'],
    argvalues=[('binary_classification_dataset',  # Binary classification
                'binary_classification_model',
                None, False),
               ('multi_class5_classification_dataset',  # Multi class classification
                'multi_class5_classification_model',
                None, False),
               ('multi_class5_classification_dataset_sparse_labels',  # Multi class classification with sparse labels
                'multi_class5_classification_model_sparse_labels',
                None, True),
               ('multi_class5_classification_dataset',  # Multi class classification with model that
                'multi_class5_classification_model_logits',  # outputs logits
                tf.nn.softmax, False),
               ('multi_class5_classification_dataset_sparse_labels',  # Multi class classification with sparse labels
                'multi_class5_classification_model_logits_sparse_labels',  # and model that outputs logits
                tf.nn.softmax, True)])
def test_classification(dataset: str, model: str,
                        outputs_to_probabilities: Optional[Callable[[Any], np.ndarray]],
                        sparse_labels: bool, request: FixtureRequest):
    """
    Test training dynamic gathering on different classification schemes, multi-class, binary, from logits and sparse
    labels
    """

    # Retrieve values
    dataset: tf.data.Dataset = request.getfixturevalue(dataset)
    model: tf.keras.Model = request.getfixturevalue(model)

    # Batch dataset
    dataset = dataset.batch(10)

    # Create callback
    datamap = tvl.learning.DataMapCallback(dataset,
                                           outputs_to_probabilities=outputs_to_probabilities,
                                           sparse_labels=sparse_labels)

    # Train
    N_EPOCHS = 5
    model.fit(dataset, epochs=N_EPOCHS, callbacks=[datamap])

    # Assert shape of gathered gold labeles probabilities are (n_samples, n_epochs)
    assert datamap.gold_labels_probabilities.shape == (100, N_EPOCHS)

    # Assert all probabilities are bound between 0 and 1
    assert datamap.gold_labels_probabilities.min() >= 0
    assert datamap.gold_labels_probabilities.max() <= 1

    # Assert training dynamics shapes
    assert datamap.confidence.shape == datamap.variability.shape == datamap.correctness.shape == (100,)


def test_exception_on_multi_class_classification(multi_label5_classification_dataset: tf.data.Dataset,
                                                 multi_class5_classification_model: tf.keras.Model):
    """ Test exception raising when gathering training dynamics on multi-label classification model """

    # Retrieve values
    dataset = multi_label5_classification_dataset
    model = multi_class5_classification_model

    # Batch dataset
    dataset = dataset.batch(10)

    # Create callback
    datamap = tvl.learning.DataMapCallback(dataset)

    # Train
    N_EPOCHS = 5
    with pytest.raises(ValueError) as excinfo:
        model.fit(dataset, epochs=N_EPOCHS, callbacks=[datamap])

    assert 'does not support multi-label classification' in str(excinfo.value)
