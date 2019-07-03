import tensorflow as tf
import numpy as np
import pytest

from tavolo.embeddings import DynamicMetaEmbedding


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

    dme = DynamicMetaEmbedding(embedding_layers, name='dynamic_meta_embedding')
    output = dme(inputs)

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

    dme = DynamicMetaEmbedding(embedding_layers, name='dynamic_meta_embedding')
    output = dme(inputs)

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

    dme = DynamicMetaEmbedding(embedding_layers, name='dynamic_meta_embedding', output_dim=output_dim)
    output = dme(inputs)

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

    dme = DynamicMetaEmbedding(embedding_layers, name='dynamic_meta_embedding')

    output = dme(inputs)

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

    simple = DynamicMetaEmbedding(embedding_layers, name='dynamic_meta_embedding')
    restored = DynamicMetaEmbedding.from_config(simple.get_config())

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
        DynamicMetaEmbedding(embedding_matrices=embedding_layers)

    assert 'Vocabulary sizes' in str(excinfo.value)
