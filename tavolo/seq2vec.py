"""
Layers mapping sequences to vectors
"""

from typing import Optional

import tensorflow as tf


class YangAttention(tf.keras.layers.Layer):
    """
    Applies attention using learned variables

    Arguments
    ---------

    - `n_units` (``int``): Attention's variables units
    - `name` (``str``): Layer name


    Input shape
    -----------

    (batch_size, time_steps, channels)


    Output shape
    ------------

    (batch_size, channels)


    Examples
    --------

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl


        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, 8, input_length=max_sequence_length),
                                     tvl.seq2vec.YangAttention()])


    References
    ----------
    `Hierarchical Attention Networks for Document Classification`_


    .. _Hierarchical Attention Networks for Document Classification: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

    """

    def __init__(self, n_units: Optional[int],
                 name: Optional[str] = 'yang_attention',
                 **kwargs):
        """
        Apply attention with learned weights.

        Input dimensions: (batch_size, time_steps, channels)
        Output dimensions: (batch_size, channels)

        Reference: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

        :param n_units: Attention's variables units
        :param name: Layer name
        """
        super(YangAttention, self).__init__(name=name, **kwargs)
        self.n_units = n_units
        self.very_small_value = (-2 ** 32 + 1)  # Used for padding to avoid attending

        # Layers
        self.omega = tf.keras.layers.Dense(self.n_units,
                                           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                           bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                           activation=tf.nn.tanh,
                                           dtype=self.dtype,
                                           name='omega')
        self.u_omega = self.add_variable('u_omega',
                                         shape=(self.n_units,),
                                         initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        # V
        v = self.omega(inputs)  # shape=(batch_size, time_steps, n_units)

        # VU
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # shape=(batch_size, time_steps)

        # Apply masking
        if mask is not None:
            # This will make sure the padded part won't be attended
            padding = tf.ones_like(vu) * self.very_small_value  # shape=(batch_size, time_steps)
            vu = tf.where(tf.equal(mask, False), padding, vu)  # shape=(batch_size, time_steps)

        # Calculate alphas
        alphas = tf.nn.softmax(vu, name='alphas')  # shape=(batch_size, time_steps)

        # Attend
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, axis=-1), axis=1)  # shape=(batch_size, channels)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['n_units'] = self.n_units

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
