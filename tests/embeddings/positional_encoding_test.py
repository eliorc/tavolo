import tensorflow as tf

from tavolo.embeddings import PositionalEncoding


def test_shapes():
    """ Test input-output shapes """

    # Inputs shape
    input_shape_3d = (56, 10, 30)

    inputs_3d = tf.random.normal(shape=input_shape_3d)

    positional_encoding = PositionalEncoding(name='positional_encoding')

    output_3d = positional_encoding(inputs_3d)

    # Assert correctness of output shapes
    assert output_3d.shape == input_shape_3d


def test_masking():
    """ Test masking support """

    # Input
    input_shape_3d = (56, 10, 30)
    inputs_3d = tf.random.normal(shape=input_shape_3d)
    mask = tf.less(tf.reduce_sum(tf.reduce_sum(inputs_3d, axis=-1, keepdims=True), axis=-1, keepdims=True), 0)
    masked_input = tf.where(tf.broadcast_to(mask, input_shape_3d), tf.zeros_like(inputs_3d), inputs_3d)

    # Layers
    masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=input_shape_3d[1:])
    positional_encoding = PositionalEncoding(name='positional_encoding')

    result = positional_encoding(masking_layer(masked_input))

    assert result.shape == input_shape_3d


def test_logic():
    """ Test logic on known input """

    # Input
    input_shape_3d = (56, 10, 30)
    inputs_3d = tf.zeros(shape=input_shape_3d, dtype=tf.float32)

    positional_encoding = PositionalEncoding(name='positional_encoding')

    # Assert output correctness
    assert tf.reduce_sum(positional_encoding(inputs_3d) - positional_encoding.positional_encoding).numpy() == 0


def test_serialization():
    """ Test layer serialization (get_config, from_config) """

    simple = PositionalEncoding()
    restored = PositionalEncoding.from_config(simple.get_config())

    assert restored.get_config() == simple.get_config()
