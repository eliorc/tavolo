import pytest
import tensorflow as tf

from tavolo.seq2seq import MultiHeadedSelfAttention


def test_shapes():
    """ Test input-output shapes """

    # Inputs shape
    input_shape_3d = (56, 10, 30)
    n_units_mh = 128

    inputs_3d = tf.random.normal(shape=input_shape_3d)

    single_self_attention = MultiHeadedSelfAttention(n_heads=1,
                                                     name='self_attention')
    multi_headed_self_attention = MultiHeadedSelfAttention(n_heads=4,
                                                           n_units=n_units_mh,
                                                           name='mh_self_attention')

    output_single, output_mh = single_self_attention(inputs_3d), multi_headed_self_attention(inputs_3d)

    # Assert correctness of output shapes
    assert output_single.shape == input_shape_3d
    assert output_mh.shape == input_shape_3d


def test_masking():
    """ Test masking support """

    # Input
    input_shape_3d = (56, 10, 30)
    inputs_3d = tf.random.normal(shape=input_shape_3d)
    mask = tf.less(tf.reduce_sum(tf.reduce_sum(inputs_3d, axis=-1, keepdims=True), axis=-1, keepdims=True), 0)
    masked_input = tf.where(tf.broadcast_to(mask, input_shape_3d), tf.zeros_like(inputs_3d), inputs_3d)

    # Layers
    masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=input_shape_3d[1:])
    multi_headed_self_attention = MultiHeadedSelfAttention(n_heads=3,
                                                           name='mh_self_attention')

    result = multi_headed_self_attention(masking_layer(masked_input))

    assert result.shape == input_shape_3d


def test_causality():
    """ Test causality """

    # Input
    input_shape_3d = (56, 10, 30)
    n_units_mh = 128
    inputs_3d = tf.random.normal(shape=input_shape_3d)

    # Layers
    multi_headed_self_attention = MultiHeadedSelfAttention(n_heads=4,
                                                           n_units=n_units_mh,
                                                           causality=True,
                                                           name='mh_self_attention')

    result = multi_headed_self_attention(inputs_3d)

    # Change last step
    inputs_3d_ = tf.concat([inputs_3d[:, :-1, :], tf.random.normal((input_shape_3d[0], 1, input_shape_3d[-1]))], axis=1)

    result_ = multi_headed_self_attention(inputs_3d_)

    # Assert last step is different but the rest not affected
    assert (result[:, :-1, :].numpy() == result_[:, :-1, :].numpy()).all()  # Without last step
    assert not (result.numpy() == result_.numpy()).all()


def test_serialization():
    """ Test layer serialization (get_config, from_config) """

    simple = MultiHeadedSelfAttention()
    restored = MultiHeadedSelfAttention.from_config(simple.get_config())

    assert restored.get_config() == simple.get_config()


def test_exceptions():
    """ Text for expected exceptions """
    # Inputs shape
    input_shape_3d = (56, 10, 30)
    n_units_mh = 99
    n_heads = 4

    inputs_3d = tf.random.normal(shape=input_shape_3d)

    multi_headed_self_attention = MultiHeadedSelfAttention(n_heads=n_heads,
                                                           n_units=n_units_mh,
                                                           name='mh_self_attention')

    # n_units % n_heads != 0, not divisible
    with pytest.raises(ValueError) as excinfo:
        multi_headed_self_attention(inputs_3d)

    assert 'n_units must be divisible by n_heads' in str(excinfo.value)
