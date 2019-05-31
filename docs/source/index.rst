Welcome to tavolo's documentation!
==================================

.. figure:: _static/logo.png
    :align: center

| `tavolo`_ aims to package together valuable modules and functionality written for `TensorFlow`_ high-level Keras API for ease of use.
| You see, the deep learning world is moving fast, and new ideas keep on coming.
| tavolo gathers implementations of these useful ideas from the community (by contribution, from `Kaggle`_ etc.)
  and makes them accessible in a single PyPI hosted package that compliments the `tf.keras`_ module.

Showcase
--------

| tavolo's API is straightforward and adopting its modules is as easy as it gets.
| In tavolo, you'll find implementations for basic layers like :ref:`layer_norm` to complex modules like the Transformer's
  :ref:`multi_headed_self_attention`.
| For example, if we wanted to head a multi-headed attention mechanism into our model, it would look something like:

.. code-block:: python3

    import tensorflow as tf
    import tavolo as tvl

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len),
        tvl.seq2seq.MultiHeadedSelfAttention(n_heads=8),  # <--- Add self attention
        tf.keras.layers.LSTM(n_lstm_units, return_sequences=True),
        tf.keras.layers.Dense(n_hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

| You are welcome continue to the :doc:`install` page, or explore the different modules available:

.. toctree::
   :hidden:
   :maxdepth: 1

   install


.. toctree::
   :maxdepth: 1
   :caption: Modules

   embeddings
   normalization
   seq2seq
   seq2vec


.. _`tavolo`: https://github.com/eliorc/tavolo
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Kaggle`: https://www.kaggle.com
.. _`tf.keras`: https://www.tensorflow.org/guide/keras