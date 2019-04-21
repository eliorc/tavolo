"""
Layers mapping sequences to vectors
"""

from typing import Optional

import tensorflow as tf


class YangAttention(tf.keras.layers.Layer):
    """
    Taken from https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    """

    def __init__(self, n_units: Optional[int],
                 name: Optional[str] = 'yang_attention',
                 **kwargs):
        """
        Apply attention with learned weights.

        Input dimensions: (batch_size, time_steps, channels)
        Output dimensions: (batch_size, channels)

        Reference: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

        :param n_units: Attention units
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
