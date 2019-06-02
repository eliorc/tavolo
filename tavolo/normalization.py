from typing import Optional

import tensorflow as tf


class LayerNorm(tf.keras.layers.Layer):
    """
    Apply layer normalization


    Arguments
    ---------

    - `epsilon` (``float``): Small number to avoid division by zero
    - `name` (``str``): Layer name


    Input shape
    -----------

    Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when
    using this layer as the first layer in a model.


    Output shape
    ------------

    Same shape as input.


    Examples
    --------

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        model = tf.keras.Sequential([SomeLayer(),
                                     tvl.normalization.LayerNorm()])  # Apply layer normalization on SomeLayer's output


    References
    ----------
    `Layer Normalization`_


    .. _Layer Normalization:
        https://arxiv.org/pdf/1607.06450
    """

    def __init__(self, epsilon: Optional[float] = 1e-8,
                 name: Optional[str] = 'layer_norm',
                 **kwargs):
        """
        :param epsilon: Small number to avoid division by zero
        :param name: Layer name
        """
        super(LayerNorm, self).__init__(name=name, **kwargs)

        self.epsilon = epsilon
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        params_shape = input_shape[-1:]

        # Initialize beta and gamma
        self.beta = self.add_variable('beta',
                                      shape=params_shape,
                                      initializer=tf.keras.initializers.zeros,
                                      dtype=self.dtype)
        self.gamma = self.add_variable('gamma',
                                       shape=params_shape,
                                       initializer=tf.keras.initializers.ones,
                                       dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             **kwargs) -> tf.Tensor:
        # Calculate mean and variance
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)  # shape=(batch_size, 1)

        # Normalize
        normalized = (inputs - mean) / ((variance + self.epsilon) ** .5)  # shape=(batch_size, channels)

        return self.gamma * normalized + self.beta  # shape=(batch_size, channels)

    def get_config(self):
        base_config = super().get_config()
        base_config['epsilon'] = self.epsilon

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
