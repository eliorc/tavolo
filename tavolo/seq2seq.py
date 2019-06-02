from typing import Optional

import tensorflow as tf


class MultiHeadedSelfAttention(tf.keras.layers.Layer):
    """
    Applies (multi headed) self attention, taken from the Transformer
    
    
    Arguments
    ---------
    
    - `n_heads` (``int``): Number of attention heads
    - `n_units` (``int``): Number of units (sum of units of all heads), defaults to the last dimension of the input
    - `dropout_rate` (``float``): Rate of outputs to drop in the range [0, 1]
    - `causality` (``bool``): Use causality (make each time point in output dependent only on previous timepoints of input)
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
                                     tvl.seq2seq.MultiHeadedSelfAttention()])


    Apply a single headed self attention

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl


        model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, 8, input_length=max_sequence_length),
                                     tvl.seq2seq.MultiHeadedSelfAttention(n_heads=1)])

    References
    ----------
    `Attention Is All You Need`_


    .. _Attention Is All You Need:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 n_heads: Optional[int] = 4,
                 n_units: Optional[int] = None,
                 dropout_rate: Optional[float] = 0.,
                 causality: Optional[bool] = False,
                 name: Optional[str] = 'multi_headed_self_attention',
                 **kwargs):
        """
        Apply multi-headed attention

        Input dimensions: (batch_size, time_steps, channels)
        Output dimensions: (batch_size, time_steps, channels)

        Reference: https://arxiv.org/abs/1706.03762

        :param n_heads: Number of attention heads
        :param n_units: Number of units (sum of units of all heads), defaults to the last dimension of the input
        :param dropout_rate: Rate of outputs to drop in the range [0, 1]
        :param causality: Use causality (make each time point in output dependent only on previous timepoints of input)
        :param name: Layer name
        """

        super(MultiHeadedSelfAttention, self).__init__(name=name, **kwargs)

        self.n_heads = n_heads
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q = None
        self.K = None
        self.V = None
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.very_small_value = (-2 ** 32 + 1)  # Used for padding to avoid attending

    def build(self, input_shape):
        # Units
        self.n_units = self.n_units or input_shape[-1]

        # Linear projections
        self.Q = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       name='Q',
                                       dtype=self.dtype)

        self.K = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       name='K',
                                       dtype=self.dtype)

        self.V = tf.keras.layers.Dense(units=self.n_units,
                                       activation=None,
                                       name='V',
                                       dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             mask: Optional[tf.Tensor] = None,
             query: Optional[tf.Tensor] = None,
             query_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = False,
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

            if query_mask is None:
                query_mask = mask

        # Linear projections
        Q = self.Q(query)  # shape=(batch_size, time_steps, n_units)
        K = self.K(inputs)  # shape=(batch_size, time_steps, n_units)
        V = self.V(inputs)  # shape=(batch_size, time_steps, n_units)

        # Split and concat
        Q = tf.concat(tf.split(Q, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        K = tf.concat(tf.split(K, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        V = tf.concat(tf.split(V, self.n_heads, axis=2),
                      axis=0)  # shape=(batch_size * n_heads, time_steps, n_units_input / n_heads)

        # Attention query
        QK = tf.matmul(Q, tf.transpose(K, perm=(0, 2, 1)))  # shape=(n_heads * batch_size, time_steps, time_steps)

        # Scale
        QK /= K.get_shape().as_list()[-1] ** 0.5  # shape=(n_heads * batch_size, time_steps, time_steps)

        # Optional key masking
        # If no mask will given a mask will be created to to represent the whole sequence minus the padding
        input_mask = mask if mask is not None else tf.sign(
            tf.abs(tf.reduce_sum(inputs, axis=-1)))  # shape=(batch_size, time_steps)
        input_mask = tf.tile(input_mask, multiples=(self.n_heads, 1))  # shape=(batch_size * n_heads, time_steps)
        input_mask = tf.tile(tf.expand_dims(input_mask, axis=1),
                             multiples=(
                                 1, tf.shape(query)[1], 1))  # shape=(batch_size * n_heads, time_steps, time_steps)
        padding = tf.ones_like(QK) * self.very_small_value  # This will make sure the padded part won't be attended
        QK = tf.where(tf.equal(input_mask, False), padding, QK)  # shape=(batch_size * n_heads, time_steps, time_steps)

        # Causality
        if self.causality:
            causality_mask = tf.ones_like(QK[0, :, :])  # shape=(time_steps, time_steps)
            causality_mask = tf.linalg.LinearOperatorLowerTriangular(
                causality_mask).to_dense()  # shape=(time_steps, time_steps)
            causality_mask = tf.tile(tf.expand_dims(  # shape=(batch_size * n_heads, time_steps, time_steps)
                causality_mask, axis=0), multiples=(tf.shape(QK)[0], 1, 1))

            padding = tf.ones_like(QK) * self.very_small_value
            QK = tf.where(tf.equal(causality_mask, False), padding,
                          QK)  # shape=(batch_size * n_heads, time_steps, time_steps)

        # Create attention weights
        alphas = tf.nn.softmax(QK)  # shape=(batch_size * n_heads, time_steps, time_steps)

        # Optional query masking
        query_mask = query_mask if query_mask is not None else tf.sign(
            tf.abs(tf.reduce_sum(query, axis=-1)))  # shape=(batch_size, time_steps)
        query_mask = tf.tile(query_mask, multiples=(self.n_heads, 1))  # shape=(batch_size * n_heads, time_steps)
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=-1), multiples=(
            1, 1, tf.shape(inputs)[1]))  # shape=(batch_size * n_heads, time_steps, time_steps)
        alphas *= tf.cast(query_mask, dtype=self.dtype)  # shape=(batch_size * n_heads, time_steps, time_steps)

        # Dropout
        alphas = self.dropout(alphas, training=training)  # shape=(batch_size * n_heads, time_steps, time_steps)

        # Attend and restore shape
        outputs = tf.matmul(alphas, V)  # shape=(batch_size * n_heads, time_steps, n_units / n_heads)
        outputs = tf.concat(tf.split(outputs, self.n_heads, axis=0),
                            axis=2)  # shape=(batch_size, time_steps, n_units)

        return outputs

    def get_config(self):
        base_config = super().get_config()
        base_config['n_heads'] = self.n_heads
        base_config['n_units'] = self.n_units
        base_config['dropout_rate'] = self.dropout_rate
        base_config['causality'] = self.causality

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
