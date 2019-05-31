import tensorflow as tf

from tavolo.normalization import LayerNorm


def test_shapes():
    """ Test input-output shapes """

    # Inputs shape
    input_shape_2d = (56, 10)
    input_shape_3d = (56, 10, 30)

    inputs_2d = tf.random.normal(shape=input_shape_2d)
    inputs_3d = tf.random.normal(shape=input_shape_3d)

    layer_norm_2d = LayerNorm(name='layer_norm_2d')
    layer_norm_3d = LayerNorm(name='layer_norm_3d')

    output_2d, output_3d = layer_norm_2d(inputs_2d), layer_norm_3d(inputs_3d)

    # Assert correctness of output shapes
    assert output_2d.shape == input_shape_2d
    assert output_3d.shape == input_shape_3d


def test_masking():
    """ Test masking support """

    # Input
    input_shape_3d = (56, 10, 30)
    inputs_3d = tf.random.normal(shape=input_shape_3d)
    mask = tf.less(tf.reduce_sum(tf.reduce_sum(inputs_3d, axis=-1), axis=-1), 0)
    masked_input = tf.where(mask, tf.zeros_like(inputs_3d), inputs_3d)

    # Layers
    masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=input_shape_3d[1:])
    layer_norm_3d = LayerNorm(name='layer_norm_3d')

    result = layer_norm_3d(masking_layer(masked_input))

    assert result.shape == input_shape_3d


def test_logic():
    """ Test logic on known input """

    # Input
    input_shape_2d = (56, 10)
    inputs_2d = tf.ones(shape=input_shape_2d)

    layer_norm_2d = LayerNorm(name='layer_norm_2d')

    # Assert output correctness
    assert tf.reduce_sum(layer_norm_2d(inputs_2d)).numpy() == 0
