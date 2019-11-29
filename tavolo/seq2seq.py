from typing import Optional

import tensorflow as tf


class MultiHeadedAttention(tf.keras.layers.Layer):
    """
    Applies (multi headed) attention, as in the Transformer
    
    
    Arguments
    ---------
    
    - `n_heads` (``int``): Number of attention heads
    - `n_units` (``int``): Number of units (sum of units of all heads), defaults to the last dimension of the input
    - `causal` (``bool``): Use causality (make each time point in output dependent only on previous timepoints of input)
    - `name` (``str``): Layer name
    
    
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
    

        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, 8, input_length=max_sequence_length),
                                     tvl.seq2seq.MultiHeadedAttention()])


    Apply a single headed self attention

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl


        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, 8, input_length=max_sequence_length),
                                     tvl.seq2seq.MultiHeadedAttention(n_heads=1)])

    .. note::

        When the intention is to apply attention using a query vector (not self attention), use the optional
        ``query`` (and ``query_mask``) argument when calling. This means that for using non-self attention
        this, you must utilize the `functional API`_ or use `model subclassing`_.


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
        Apply multi-headed attention

        Input dimensions: (batch_size, time_steps, channels)
        Output dimensions: (batch_size, time_steps, channels)

        Reference: https://arxiv.org/abs/1706.03762

        :param n_heads: Number of attention heads
        :param n_units: Number of units (sum of units of all heads), defaults to the last dimension of the input
        :param causal: Use causality (make each time point in output dependent only on previous timepoints of input)
        :param name: Layer name
        """

        super().__init__(name=name, **kwargs)

        self.n_heads = n_heads
        self.n_units = n_units
        self.causal = causal
        self.Q = None
        self.K = None
        self.V = None
        self.output_projection = None
        self.very_small_value = (-2 ** 32 + 1)  # Used for padding to avoid attending

    def build(self, input_shape):
        # Units
        channels = input_shape[-1]
        self.n_units = self.n_units or channels

        # Test units - n_heads validity
        if self.n_units % self.n_heads != 0:
            raise ValueError('n_units must be divisible by n_heads')

        # Linear projections
        self.Q = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       use_bias=False,
                                       name='Q',
                                       dtype=self.dtype)

        self.K = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       use_bias=False,
                                       name='K',
                                       dtype=self.dtype)

        self.V = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       use_bias=False,
                                       name='V',
                                       dtype=self.dtype)

        self.attention = tf.keras.layers.Attention(use_scale=True,
                                                   causal=self.causal)

        self.output_projection = tf.keras.layers.Dense(units=channels,
                                                       activation=None,
                                                       use_bias=False,
                                                       name='output_projection',
                                                       dtype=self.dtype)

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             mask: Optional[tf.Tensor] = None,
             query: Optional[tf.Tensor] = None,
             query_mask: Optional[tf.Tensor] = None,
             training: bool = False,
             **kwargs) -> tf.Tensor:
        """

        :param inputs: Inputs to be used as K and V (ex. word embeddings)
        :param mask: Mask from the previous layer
        :param query: Query to be used as Q
        :param query_mask: Mask for the query
        :param training: Is training
        """

        if query is None:
            query = inputs  # Self attention
            query_mask = mask

        # Linear projections
        Q = self.Q(query)  # shape=(batch_size, time_steps, n_units)
        K = self.K(inputs)  # shape=(batch_size, time_steps, n_units)
        V = self.V(inputs)  # shape=(batch_size, time_steps, n_units)

        # Split and concat, for parallel execution
        Q = tf.concat(tf.split(Q, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        K = tf.concat(tf.split(K, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        V = tf.concat(tf.split(V, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units_input / n_heads)
        attention_mask = list()
        if query_mask is not None:
            query_mask = tf.tile(query_mask, multiples=(self.n_heads, 1))  # shape=(batch_size * n_heads, time_steps)
            attention_mask.append(query_mask)
        if mask is not None:
            mask = tf.tile(mask, multiples=(self.n_heads, 1))  # shape=(batch_size * n_heads, time_steps)
            attention_mask.append(mask)

        # Attention query
        attended = self.attention([Q, V, K],
                                  mask=attention_mask)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)

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
