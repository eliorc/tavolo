import pytest
import tensorflow as tf

from tavolo.seq2seq import MultiHeadedAttention


def test_shapes():
    """ Test input-output shapes """

    # Inputs shape
    input_shape_3d = (56, 10, 30)
    n_units_mh = 128

    inputs_3d = tf.random.normal(shape=input_shape_3d)

    single_self_attention = MultiHeadedAttention(n_heads=1,
                                                 name='self_attention')
    multi_headed_self_attention = MultiHeadedAttention(n_heads=4,
                                                       n_units=n_units_mh,
                                                       name='mh_self_attention')

    output_single, output_mh = single_self_attention([inputs_3d, inputs_3d]), multi_headed_self_attention(
        [inputs_3d, inputs_3d])

    # Assert correctness of output shapes
    assert output_single.shape == input_shape_3d
    assert output_mh.shape == input_shape_3d


def test_query_key_value():
    """ Test the ability to use query, key and value separately """
    # Inputs shape
    input_shape_3d = (56, 10, 30)
    n_units_mh = 128

    inputs_3d = tf.random.normal(shape=input_shape_3d)

    multi_headed_attention = MultiHeadedAttention(n_heads=4,
                                                  n_units=n_units_mh,
                                                  name='mh_attention')

    output_self, output_non_self = multi_headed_attention([inputs_3d, inputs_3d]), \
                                   multi_headed_attention([inputs_3d, inputs_3d, inputs_3d])

    assert tf.reduce_all(tf.math.equal(output_self, output_non_self))


def test_masking():
    """ Test masking support """

    # Input
    input_shape_3d = (56, 10, 30)
    inputs_3d = tf.random.normal(shape=input_shape_3d)
    mask = tf.less(tf.reduce_sum(tf.reduce_sum(inputs_3d, axis=-1, keepdims=True), axis=-1), 0)

    # Layers
    multi_headed_self_attention = MultiHeadedAttention(n_heads=3,
                                                       name='mh_self_attention')

    result = multi_headed_self_attention([inputs_3d, inputs_3d], mask=[mask, mask])

    assert result.shape == input_shape_3d


def test_causality():
    """ Test causality """

    # Input
    input_shape_3d = (56, 10, 30)
    n_units_mh = 128
    inputs_3d = tf.random.normal(shape=input_shape_3d)

    # Layers
    multi_headed_self_attention = MultiHeadedAttention(n_heads=4,
                                                       n_units=n_units_mh,
                                                       causal=True,
                                                       name='mh_self_attention')

    result = multi_headed_self_attention([inputs_3d, inputs_3d])

    # Change last step
    inputs_3d_ = tf.concat([inputs_3d[:, :-1, :], tf.random.normal((input_shape_3d[0], 1, input_shape_3d[-1]))], axis=1)

    result_ = multi_headed_self_attention([inputs_3d_, inputs_3d_])

    # Assert last step is different but the rest not affected
    assert (result[:, :-1, :].numpy() == result_[:, :-1, :].numpy()).all()  # Without last step
    assert not (result.numpy() == result_.numpy()).all()


def test_serialization():
    """ Test layer serialization (get_config, from_config) """

    simple = MultiHeadedAttention()
    restored = MultiHeadedAttention.from_config(simple.get_config())

    assert restored.get_config() == simple.get_config()
