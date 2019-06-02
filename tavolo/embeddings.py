from typing import Optional

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Create a positional encoding layer, usually added on top of an embedding layer.
    Embeds information about the position of the elements using the formula

    .. math::

        PE[pos,2i]=sin\\left(\\frac{pos}{normalize\\_factor^{\\frac{2i}{embedding\\_dim}}}\\right)

        PE[pos,2i+1]=cos\\left(\\frac{pos}{normalize\\_factor^{\\frac{2i}{embedding\\_dim}}}\\right)


    The resulting embedding gets added (point-wise) to the input.


    Arguments
    ---------

    - `max_sequence_length` (``int``): Maximum sequence length of input
    - `embedding_dim` (``int``): Dimensionality of the of the input's last dimension
    - `normalize_factor` (``float``): Normalize factor
    - `name` (``str``): Layer name


    Input shape
    -----------

    (batch_size, time_steps, channels) where time_steps equals to the ``max_sequence_length`` and channels to ``embedding_dim``


    Output shape
    ------------

    Same shape as input.


    Examples
    --------

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, 8, input_length=max_sequence_length),
                                     tvl.embeddings.PositionalEncoding(max_sequence_length=max_sequence_length,
                                                                       embedding_dim=8)])  # Add positional encoding


    References
    ----------
    `Attention Is All You Need`_


    .. _Attention Is All You Need:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 max_sequence_length: int,
                 embedding_dim: int,
                 normalize_factor: Optional[float] = 10000,
                 name: Optional[str] = 'positional_encoding',
                 **kwargs):
        """
        :param max_sequence_length: Maximum sequence length of input
        :param embedding_dim: Dimensionality of the of the input's last dimension
        :param normalize_factor: Normalize factor
        :param name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.normalize_factor = normalize_factor

        # Error checking
        if max_sequence_length < 1:
            raise ValueError(
                'max_sequence_length must be greater than zero. (value provided {})'.format(max_sequence_length))

        if embedding_dim < 1:
            raise ValueError(
                'embedding_dim must be greater than zero. (value provided {})'.format(max_sequence_length))

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

    def call(self, inputs,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:

        output = inputs + self.positional_encoding
        if mask is not None:
            output = tf.where(tf.tile(tf.expand_dims(mask, axis=-1), multiples=[1, 1, inputs.shape[-1]]), output,
                              inputs)

        return output  # shape=(batch_size, time_steps, channels)

    def get_config(self):
        base_config = super().get_config()
        base_config['max_sequence_length'] = self.max_sequence_length
        base_config['embedding_dim'] = self.embedding_dim
        base_config['normalize_factor'] = self.normalize_factor

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
