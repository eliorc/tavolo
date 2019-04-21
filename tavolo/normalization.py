"""
Normalization techniques
"""

from typing import Optional

import tensorflow as tf


class LayerNorm(tf.keras.layers.Layer):

    def __init__(self, epsilon: Optional[float] = 1e-8,
                 name: Optional[str] = 'layer_norm',
                 **kwargs):
        """
        Apply layer norm

        Input dimensions: (batch_size, channels)
        Output dimensions: (batch_size, channels)

        Reference: https://arxiv.org/abs/1607.06450

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
        mean, variance = tf.nn.moments(inputs, axes=-1, keep_dims=True)  # shape=(batch_size, 1)

        # Normalize
        normalized = (inputs - mean) / ((variance + self.epsilon) ** .5)  # shape=(batch_size, channels)

        return self.gamma * normalized + self.beta  # shape=(batch_size, channels)
