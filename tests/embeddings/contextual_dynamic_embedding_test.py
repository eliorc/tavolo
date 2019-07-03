import tensorflow as tf
import numpy as np
import pytest

from tavolo.embeddings import ContextualDynamicMetaEmbedding


def test_shapes():
    """ Test input-output shapes """

    # -------- STANDARD CASE --------

    # Inputs shape
    input_shape = (56, 10)
    embedding_matrices_shapes = [(100, 30), (100, 30)]

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False) for e in embedding_matrices]

    cdme = ContextualDynamicMetaEmbedding(embedding_layers, name='contextual_dynamic_meta_embedding')
    output = cdme(inputs)

    channel_dim = embedding_matrices_shapes[0][1]

    # Assert correctness of output shapes
    assert output.shape == (input_shape[0], input_shape[1], channel_dim)

    # -------- MULTIPLE EMBEDDING DIMS --------

    embedding_matrices_shapes = [(100, 25), (100, 30)]

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False) for e in embedding_matrices]

    cdme = ContextualDynamicMetaEmbedding(embedding_layers, name='contextual_dynamic_meta_embedding')
    output = cdme(inputs)

    # Figure out default output dimension (minimum of provided embeddings)
    channel_dim = min(s[1] for s in embedding_matrices_shapes)

    # Assert correctness of output shapes
    assert output.shape == (input_shape[0], input_shape[1], channel_dim)

    # -------- USER DEFINED OUTPUT DIM --------

    embedding_matrices_shapes = [(100, 25), (100, 30)]
    output_dim = 15

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False) for e in embedding_matrices]

    cdme = ContextualDynamicMetaEmbedding(embedding_layers, name='contextual_dynamic_meta_embedding',
                                          output_dim=output_dim)
    output = cdme(inputs)

    # Assert correctness of output shapes
    assert output.shape == (input_shape[0], input_shape[1], output_dim)

    # -------- USER DEFINED OUTPUT DIM AND LSTM UNITS --------

    embedding_matrices_shapes = [(100, 25), (100, 30)]
    output_dim = 15
    n_lstm_units = 24

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False) for e in embedding_matrices]

    cdme = ContextualDynamicMetaEmbedding(embedding_layers,
                                          n_lstm_units=n_lstm_units,
                                          output_dim=output_dim,
                                          name='contextual_dynamic_meta_embedding')
    output = cdme(inputs)

    # Assert correctness of output shapes
    assert output.shape == (input_shape[0], input_shape[1], output_dim)


def test_masking():
    """ Test masking support """

    # Inputs shape
    input_shape = (56, 10)
    embedding_matrices_shapes = [(100, 30), (100, 30)]

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False,
                                                  mask_zero=True) for e in embedding_matrices]

    # Make sure we have a masked item
    inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')
    while not bool(tf.reduce_any(tf.equal(inputs, 0))):
        inputs = tf.random.uniform(shape=input_shape, maxval=100, dtype='int32')

    cdme = ContextualDynamicMetaEmbedding(embedding_layers, name='contextual_dynamic_meta_embedding')

    output = cdme(inputs)

    channel_dim = embedding_matrices_shapes[0][1]

    # Assert correctness of output shapes
    assert output.shape == (input_shape[0], input_shape[1], channel_dim)


def test_serialization():
    """ Test layer serialization (get_config, from_config) """

    # Inputs shape
    input_shape = (56, 10)
    embedding_matrices_shapes = [(100, 30), (100, 30)]

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False,
                                                  mask_zero=True) for e in embedding_matrices]

    simple = ContextualDynamicMetaEmbedding(embedding_layers, name='contextual_dynamic_meta_embedding')
    restored = ContextualDynamicMetaEmbedding.from_config(simple.get_config())

    assert restored.get_config() == simple.get_config()


def test_exceptions():
    """ Text for expected exceptions """

    # Inputs shape
    input_shape = (56, 10)
    embedding_matrices_shapes = [(98, 30), (100, 30)]

    # Create embedding matrices
    embedding_matrices = [np.random.normal(size=shape) for shape in embedding_matrices_shapes]

    # Create embedding layers
    embedding_layers = [tf.keras.layers.Embedding(e.shape[0],
                                                  e.shape[1],
                                                  embeddings_initializer=tf.keras.initializers.Constant(e),
                                                  trainable=False,
                                                  mask_zero=True) for e in embedding_matrices]

    # Sequence length lower than 1
    with pytest.raises(ValueError) as excinfo:
        ContextualDynamicMetaEmbedding(embedding_matrices=embedding_layers)

    assert 'Vocabulary sizes' in str(excinfo.value)
