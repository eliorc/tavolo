from typing import Optional, List

import tensorflow as tf


class MultiHeadedAttention(tf.keras.layers.Layer):
    """
    Applies (multi headed) attention, as in the Transformer
    
    
    Arguments
    ---------
    
    - `n_heads` (``int``): Number of attention heads
    - `n_units` (``int``): Number of units per head, defaults to the last dimension of the input
    - `causal` (``bool``): Use causality (make each time point in output dependent only on previous time points of input)
    - `name` (``str``): Layer name

    ``call`` Arguments
    ------------------

    - ``inputs`` (``List[tf.Tensor]``): List of the following tensors

     - query: Query Tensor of shape [batch_size, Tq, dim]
     - value: Value Tensor of shape [batch_size, Tv, dim].
     - key: Optional key Tensor of shape [batch_size, Tv, dim].
            If not given, will use value for both key and value, which is the most common case

    - ``mask`` (``List[tf.Tensor]``): List of the following tensors

     - query_mask: A boolean mask Tensor of shape [batch_size, Tq].
                   If given, the output will be zero at the positions where mask==False
     - value_mask: A boolean mask Tensor of shape [batch_size, Tv].
                   If given, will apply the mask such that values at positions where mask==False do not
                   contribute to the result
    
    
    Input shape
    -----------
    
    (batch_size, time_steps, channels)
    
    
    Output shape
    ------------
    
    Same shape as input.
    
    
    Examples
    --------

    Apply a 4 headed (default) self attention
    
    .. code-block:: python3
    
        import tensorflow as tf
        import tavolo as tvl

        # Inputs
        inputs = tf.keras.Input(shape=(max_seq_length,), dtype='int32')

        # Embedding lookup
        embedding_layer = tf.keras.layers.Embedding(max_tokens, dimension)
        embedded = embedding_layer(inputs)

        # Apply multi headed self attention
        mh_attention = tvl.seq2seq.MultiHeadedAttention()
        attended = mh_attention([embedded, embedded])


    Apply a 4 headed attention, using a query vector and masking

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        # Inputs
        query_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32')
        value_input = tf.keras.Input(shape=(max_seq_length,), dtype='int32')

        # Embedding lookup
        embedding_layer = tf.keras.layers.Embedding(max_tokens, dimension, mask_zero=True)
        embedded_query = embedding_layer(query_input)
        embedded_value = embedding_layer(value_input)

        # Masks
        query_mask = embedding_layer.compute_mask(query_input)
        value_mask = embedding_layer.compute_mask(value_input)

        # Apply multi headed self attention
        mh_attention = tvl.seq2seq.MultiHeadedAttention()
        attended = mh_attention([embedded_query, embedded_value], mask=[query_mask, value_mask])

    .. note::

        Since the query and value should be passed separately, it is recommended to use the `functional API`_ or
        `model subclassing`_ to use this layer.


    .. _`functional API`:
        https://www.tensorflow.org/guide/keras/functional

    .. _`model subclassing`:
        https://www.tensorflow.org/guide/keras/custom_layers_and_models#building_models


    References
    ----------
    `Attention Is All You Need`_


    .. _Attention Is All You Need:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 n_heads: int = 4,
                 n_units: Optional[int] = None,
                 causal: bool = False,
                 name: str = 'multi_headed_attention',
                 **kwargs):
        """

        :param n_heads: Number of attention heads
        :param n_units: Number of units per head, defaults to the last dimension of the input
        :param causal: Use causality (make each time point in output dependent only on previous time points of input)
        :param name: Layer name
        """

        super().__init__(name=name, **kwargs)

        self.n_heads = n_heads
        self.n_units = n_units
        self.causal = causal
        self.Q = None
        self.K = None
        self.V = None
        self.attention = None
        self.output_projection = None
        self.very_small_value = (-2 ** 32 + 1)  # Used for padding to avoid attending

    def build(self, input_shape):
        # Units
        channels = input_shape[0][-1]
        self.n_units = self.n_units or channels

        # Linear projections
        self.Q = tf.keras.layers.Dense(units=self.n_units * self.n_heads,
                                       activation=None,
                                       use_bias=False,
                                       name='Q',
                                       dtype=self.dtype)

        self.K = tf.keras.layers.Dense(units=self.n_units * self.n_heads,
                                       activation=None,
                                       use_bias=False,
                                       name='K',
                                       dtype=self.dtype)

        self.V = tf.keras.layers.Dense(units=self.n_units * self.n_heads,
                                       activation=None,
                                       use_bias=False,
                                       name='V',
                                       dtype=self.dtype)

        self.attention = tf.keras.layers.Attention(use_scale=True,
                                                   causal=self.causal,
                                                   name='attention',
                                                   dtype=self.dtype)

        self.output_projection = tf.keras.layers.Dense(units=channels,
                                                       activation=None,
                                                       use_bias=False,
                                                       name='output_projection',
                                                       dtype=self.dtype)

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        return mask[0]

    def call(self, inputs,
             mask: Optional[List[tf.Tensor]] = None,
             training: bool = False,
             **kwargs) -> tf.Tensor:
        """
        :param inputs: List of the following tensors:

         - query: Query Tensor of shape [batch_size, Tq, dim]
         - value: Value Tensor of shape [batch_size, Tv, dim].
         - key: Optional key Tensor of shape [batch_size, Tv, dim].
                If not given, will use value for both key and value, which is the most common case

        :param mask: List of the following tensors:

         - query_mask: A boolean mask Tensor of shape [batch_size, Tq].
                       If given, the output will be zero at the positions where mask==False
         - value_mask: A boolean mask Tensor of shape [batch_size, Tv].
                       If given, will apply the mask such that values at positions where mask==False do not
                       contribute to the result

        :param training: Is training
        """

        # Unpack inputs
        query = inputs[0]
        value = inputs[1]
        key = inputs[2] if len(inputs) > 2 else value

        # Linear projections
        Q = self.Q(query)  # shape=(batch_size, time_steps, n_units)
        K = self.K(key)  # shape=(batch_size, time_steps, n_units)
        V = self.V(value)  # shape=(batch_size, time_steps, n_units)

        # Split and concat, for parallel execution
        Q = tf.concat(tf.split(Q, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        K = tf.concat(tf.split(K, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        V = tf.concat(tf.split(V, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units_input / n_heads)

        # Attention query
        if mask is None or len(mask) == 0:
            attended = self.attention([Q, V, K])  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        else:
            tiled_mask = [tf.tile(m, multiples=(self.n_heads, 1)) for m in
                          mask]  # shape=(batch_size * n_heads, time_steps)
            attended = self.attention([Q, V, K],
                                      mask=tiled_mask)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)

        # Restore original shape
        outputs = tf.concat(tf.split(attended, self.n_heads, axis=0),
                            axis=2)  # shape=(batch_size, time_steps, n_units)

        # Project output
        outputs = self.output_projection(outputs)  # shape=(batch_size, time_steps, channels)

        return outputs

    def get_config(self):
        base_config = super().get_config()
        base_config['n_heads'] = self.n_heads
        base_config['n_units'] = self.n_units
        base_config['causal'] = self.causal

        return base_config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
