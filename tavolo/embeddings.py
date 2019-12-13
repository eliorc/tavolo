from typing import Optional, List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


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
                 normalize_factor: float = 10000,
                 name: str = 'positional_encoding',
                 **kwargs):
        """

        :param max_sequence_length: Maximum sequence length of input
        :param embedding_dim: Dimensionality of the of the input's last dimension
        :param normalize_factor: Normalize factor
        :param name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.normalize_factor = normalize_factor
        self.positional_encoding = None

    def build(self, input_shape):
        max_sequence_length, embedding_dim = input_shape[-2:]

        # First part of the PE function: sin and cos argument
        self.positional_encoding = np.array([
            [pos / np.power(self.normalize_factor, 2. * i / embedding_dim) for i in range(embedding_dim)]
            for pos in range(max_sequence_length)])

        # Second part, apply the cosine to even columns and sin to odds.
        self.positional_encoding[:, 0::2] = np.sin(self.positional_encoding[:, 0::2])
        self.positional_encoding[:, 1::2] = np.cos(self.positional_encoding[:, 1::2])

        self.positional_encoding = self.add_weight(
            'embedding_matrix',
            shape=self.positional_encoding.shape,
            initializer=tf.keras.initializers.Constant(self.positional_encoding),
            trainable=False,
            dtype=self.dtype)

        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        output = inputs + self.positional_encoding  # shape=(batch_size, time_steps, channels)

        if mask is not None:
            output = tf.where(tf.tile(tf.expand_dims(mask, axis=-1), multiples=[1, 1, inputs.shape[-1]]), output,
                              inputs)  # shape=(batch_size, time_steps, channels)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['normalize_factor'] = self.normalize_factor

        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DynamicMetaEmbedding(tf.keras.layers.Layer):
    """
    Applies learned attention to different sets of embeddings matrices per token, to mix separate token
    representations into a joined one. Self attention is word-dependent, meaning each word's representation in the output
    is only dependent on the word's original embeddings in the given matrices, and the attention vector.


    Arguments
    ---------

    - `embedding_matrices` (``List[np.ndarray]``): List of embedding matrices
    - `output_dim` (``int``): Dimension of the output embedding
    - `mask_zero` (``bool``): Whether or not the input value 0 is a special "padding" value that should be masked out
    - `input_length` (``Optional[int]``): Parameter to be passed into internal ``tf.keras.layers.Embedding`` matrices
    - `name` (``str``): Layer name


    Input shape
    -----------

    (batch_size, time_steps)


    Output shape
    ------------

    (batch_size, time_steps, output_dim)


    Examples
    --------

    Create Dynamic Meta Embeddings using 2 separate embedding matrices. Notice it is the user's responsibility to make sure
    all the arguments needed in the embedding lookup are passed to the ``tf.keras.layers.Embedding`` constructors (like ``trainable=False``).

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        w2v_embedding = np.array(...)  # Pre-trained embedding matrix

        glove_embedding = np.array(...)  # Pre-trained embedding matrix

        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
                                     tvl.embeddings.DynamicMetaEmbedding([w2v_embedding, glove_embedding],
                                                                         input_length=MAX_SEQUENCE_LENGTH)])  # Use DME embeddings

    Using the same example as above, it is possible to define the output's channel size

    .. code-block:: python3

        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
                                     tvl.embeddings.DynamicMetaEmbedding([w2v_embedding, glove_embedding],
                                                                         input_length=MAX_SEQUENCE_LENGTH,
                                                                         output_dim=200)])


    References
    ----------
    `Dynamic Meta-Embeddings for Improved Sentence Representations`_


    .. _`Dynamic Meta-Embeddings for Improved Sentence Representations`:
        https://arxiv.org/abs/1804.07983
    """

    def __init__(self,
                 embedding_matrices: List[np.ndarray],
                 output_dim: Optional[int] = None,
                 mask_zero: bool = False,
                 input_length: Optional[int] = None,
                 name: str = 'dynamic_meta_embedding',
                 **kwargs):
        """

        :param embedding_matrices: List of embedding matrices
        :param output_dim: Dimension of the output embedding
        :param mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out
        :param input_length: Parameter to be passed into internal ``tf.keras.layers.Embedding`` matrices
        :param name: Layer name
        """
        super().__init__(name=name, **kwargs)

        self.mask_zero = mask_zero
        self.input_length = input_length
        self.base_matrices_shapes = [e.shape for e in embedding_matrices]

        # Validate all the embedding matrices have the same vocabulary size
        if not len(set((e.shape[0] for e in embedding_matrices))) == 1:
            raise ValueError('Vocabulary sizes (first dimension) of all embedding matrices must match')
        if not set((e.ndim for e in embedding_matrices)) == {2}:
            raise ValueError('All embedding matrices should have only 2 dimensions')

        # If no output_dim is supplied, use the maximum dimension from the given matrices
        self.output_dim = output_dim or min([e.shape[1] for e in embedding_matrices])

        self.embedding_matrices = [tf.keras.layers.Embedding(input_dim=e.shape[0],
                                                             output_dim=e.shape[1],
                                                             embeddings_initializer=tf.keras.initializers.Constant(e),
                                                             input_length=input_length,
                                                             name='embedding_matrix_{}'.format(i))
                                   for i, e in enumerate(embedding_matrices)]
        self.n_embeddings = len(self.embedding_matrices)

        self.projections = [tf.keras.layers.Dense(units=self.output_dim,
                                                  activation=None,
                                                  name='projection_{}'.format(i),
                                                  dtype=self.dtype) for i, e in enumerate(self.embedding_matrices)]

        self.attention = tf.keras.layers.Dense(units=1,
                                               activation=None,
                                               name='attention',
                                               dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    def call(self, inputs,
             **kwargs) -> tf.Tensor:

        batch_size, time_steps = inputs.shape[:2]

        # Embedding lookup
        embedded = [e(inputs) for e in self.embedding_matrices]  # List of shape=(batch_size, time_steps, channels_i)

        # Projection
        projected = tf.reshape(tf.concat([p(e) for p, e in zip(self.projections, embedded)], axis=-1),
                               # Project embeddings
                               shape=(batch_size, time_steps, -1, self.output_dim),
                               name='projected')  # shape=(batch_size, time_steps, n_embeddings, output_dim)

        # Calculate attention coefficients
        alphas = self.attention(projected)  # shape=(batch_size, time_steps, n_embeddings, 1)
        alphas = tf.nn.softmax(alphas, axis=-2)  # shape=(batch_size, time_steps, n_embeddings, 1)

        # Attend
        output = tf.squeeze(tf.matmul(
            tf.transpose(projected, perm=[0, 1, 3, 2]), alphas),  # Attending
            name='output')  # shape=(batch_size, time_steps, output_dim)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['base_matrices_shapes'] = self.base_matrices_shapes
        base_config['output_dim'] = self.output_dim
        base_config['mask_zero'] = self.mask_zero
        base_config['input_length'] = self.input_length

        return base_config

    @classmethod
    def from_config(cls, config: dict):
        initial_matrices = [np.zeros(shape=s) for s in config.pop('base_matrices_shapes')]
        return cls(embedding_matrices=initial_matrices, **config)


class ContextualDynamicMetaEmbedding(tf.keras.layers.Layer):
    """
    Applies learned attention to different sets of embeddings matrices per token, to mix separate token
    representations into a joined one. Self attention is context-dependent, meaning each word's representation in the output
    is only dependent on the sentence's original embeddings in the given matrices, and the attention vector.
    The context is generated by a BiLSTM.


    Arguments
    ---------

    - `embedding_matrices` (``List[np.ndarray]``): List of embedding matrices
    - `output_dim` (``int``): Dimension of the output embedding
    - `mask_zero` (``bool``): Whether or not the input value 0 is a special "padding" value that should be masked out
    - `input_length` (``Optional[int]``): Parameter to be passed into internal ``tf.keras.layers.Embedding`` matrices
    - `n_lstm_units` (``int``): Number of units in each LSTM, (notated as `m` in the original article)
    - `name` (``str``): Layer name


    Input shape
    -----------

    (batch_size, time_steps)


    Output shape
    ------------

    (batch_size, time_steps, output_dim)


    Examples
    --------

    Create Dynamic Meta Embeddings using 2 separate embedding matrices. Notice it is the user's responsibility to make sure
    all the arguments needed in the embedding lookup are passed to the ``tf.keras.layers.Embedding`` constructors (like ``trainable=False``).

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        w2v_embedding = np.array(...)  # Pre-trained embedding matrix

        glove_embedding = np.array(...)  # Pre-trained embedding matrix

        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
                                     tvl.embeddings.DynamicMetaEmbedding([w2v_embedding, glove_embedding],
                                                                         input_length=MAX_SEQUENCE_LENGTH)])  # Use CDME embeddings

    Using the same example as above, it is possible to define the output's channel size and number of units in each LSTM

    .. code-block:: python3

        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),
                                     tvl.embeddings.DynamicMetaEmbedding([w2v_embedding, glove_embedding],
                                                                          input_length=MAX_SEQUENCE_LENGTH,
                                                                          n_lstm_units=128, output_dim=200)])

    References
    ----------
    `Dynamic Meta-Embeddings for Improved Sentence Representations`_


    .. _`Dynamic Meta-Embeddings for Improved Sentence Representations`:
        https://arxiv.org/abs/1804.07983
    """

    def __init__(self,
                 embedding_matrices: List[np.ndarray],
                 output_dim: Optional[int] = None,
                 mask_zero: bool = False,
                 input_length: Optional[int] = None,
                 n_lstm_units: int = 2,
                 name: str = 'contextual_dynamic_meta_embedding',
                 **kwargs):
        """
        :param embedding_matrices: List of embedding matrices
        :param output_dim: Dimension of the output embedding
        :param mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out
        :param input_length: Parameter to be passed into internal ``tf.keras.layers.Embedding`` matrices
        :param n_lstm_units: Number of units in each LSTM, (notated as `m` in the original article)
        :param name: Layer name
        """

        super().__init__(name=name, **kwargs)

        self.mask_zero = mask_zero
        self.input_length = input_length
        self.n_lstm_units = n_lstm_units
        self.base_matrices_shapes = [e.shape for e in embedding_matrices]

        # Validate all the embedding matrices have the same vocabulary size
        if not len(set((e.shape[0] for e in embedding_matrices))) == 1:
            raise ValueError('Vocabulary sizes (first dimension) of all embedding matrices must match')
        if not set((e.ndim for e in embedding_matrices)) == {2}:
            raise ValueError('All embedding matrices should have only 2 dimensions')

        # If no output_dim is supplied, use the maximum dimension from the given matrices
        self.output_dim = output_dim or min([e.shape[1] for e in embedding_matrices])

        self.embedding_matrices = [tf.keras.layers.Embedding(input_dim=e.shape[0],
                                                             output_dim=e.shape[1],
                                                             embeddings_initializer=tf.keras.initializers.Constant(e),
                                                             input_length=input_length,
                                                             name='embedding_matrix_{}'.format(i))
                                   for i, e in enumerate(embedding_matrices)]
        self.n_embeddings = len(self.embedding_matrices)

        self.projections = [tf.keras.layers.Dense(units=self.output_dim,
                                                  activation=None,
                                                  name='projection_{}'.format(i),
                                                  dtype=self.dtype) for i, e in enumerate(self.embedding_matrices)]

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=self.n_lstm_units, return_sequences=True),
            name='bilstm',
            dtype=self.dtype)

        self.attention = tf.keras.layers.Dense(units=1,
                                               activation=None,
                                               name='attention',
                                               dtype=self.dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    def call(self, inputs,
             **kwargs) -> tf.Tensor:
        batch_size, time_steps = inputs.shape[:2]

        # Embedding lookup
        embedded = [e(inputs) for e in self.embedding_matrices]  # List of shape=(batch_size, time_steps, channels_i)

        # Projection
        projected = tf.reshape(tf.concat([p(e) for p, e in zip(self.projections, embedded)], axis=-1),
                               # Project embeddings
                               shape=(batch_size, time_steps, -1, self.output_dim),
                               name='projected')  # shape=(batch_size, time_steps, n_embeddings, output_dim)

        # Contextualize
        context = self.bilstm(
            tf.reshape(projected, shape=(batch_size * self.n_embeddings, time_steps,
                                         self.output_dim)))  # shape=(batch_size * n_embeddings, time_steps, n_lstm_units*2)
        context = tf.reshape(context, shape=(batch_size, time_steps, self.n_embeddings,
                                             self.n_lstm_units * 2))  # shape=(batch_size, time_steps, n_embeddings, n_lstm_units*2)

        # Calculate attention coefficients
        alphas = self.attention(context)  # shape=(batch_size, time_steps, n_embeddings, 1)
        alphas = tf.nn.softmax(alphas, axis=-2)  # shape=(batch_size, time_steps, n_embeddings, 1)

        # Attend
        output = tf.squeeze(tf.matmul(
            tf.transpose(projected, perm=[0, 1, 3, 2]), alphas),  # Attending
            name='output')  # shape=(batch_size, time_steps, output_dim)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['base_matrices_shapes'] = self.base_matrices_shapes
        base_config['output_dim'] = self.output_dim
        base_config['mask_zero'] = self.mask_zero
        base_config['input_length'] = self.input_length
        base_config['n_lstm_units'] = self.n_lstm_units

        return base_config

    @classmethod
    def from_config(cls, config: dict):
        initial_matrices = [np.zeros(shape=s) for s in config.pop('base_matrices_shapes')]
        return cls(embedding_matrices=initial_matrices, **config)
