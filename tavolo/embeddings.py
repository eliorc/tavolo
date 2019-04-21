"""
Layers applied to embeddings
"""

from typing import Optional

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self,
                 max_sequence_length: int,
                 embedding_dim: int,
                 normalize_factor: Optional[float] = 10000,
                 name: Optional[str] = 'positional_encoding',
                 **kwargs):
        """
        Adds positional encoding using sin and cos

        Input dimensions: (batch_size, time_steps, channels)
        Output dimensions: (batch_size, time_steps, channels)

        Reference: Reference: https://arxiv.org/abs/1706.03762

        :param max_sequence_length: Maximum allowed sequence length
        :param embedding_dim: Embedding dimension
        :param normalize_factor: Normalize factor
        :param name: Layer name
        """
        super(PositionalEncoding, self).__init__(name=name, **kwargs)

        # First part of the PE function: sin and cos argument
        self.positional_encoding = np.array([
            [pos / np.power(normalize_factor, 2. * i / embedding_dim) for i in range(embedding_dim)]
            for pos in range(max_sequence_length)])

        # Second part, apply the cosine to even columns and sin to odds.
        self.positional_encoding[:, 0::2] = np.sin(self.positional_encoding[:, 0::2])
        self.positional_encoding[:, 1::2] = np.cos(self.positional_encoding[:, 1::2])

        self.positional_encoding = self.add_variable(
            'embedding_matrix',
            shape=self.positional_encoding.shape,
            initializer=tf.keras.initializers.Constant(self.positional_encoding),
            trainable=False,
            dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, input, **kwargs) -> tf.Tensor:
        return input + self.positional_encoding  # shape=(batch_size, time_steps, channels)
