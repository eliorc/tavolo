.. raw:: html

    <div align="center">
      <img src="docs/source/_static/logo.png"><br><br>
    </div>

------------

Tavolo
======

| `tavolo`_ aims to package together valuable modules and functionality written for `TensorFlow`_ high-level Keras API for ease of use.
| You see, the deep learning world is moving fast, and new ideas keep on coming.
| tavolo gathers implementations of these useful ideas from the community (by contribution, from `Kaggle`_ etc.)
  and makes them accessible in a single PyPI hosted package that compliments the `tf.keras`_ module.

Documentation
-------------

| `Tavolo documentation`_

.. _`Tavolo documentation`: https://tavolo.readthedocs.io/

Showcase
--------

.. TODO - Add LayerNorm and MultiHeadedSelfAttention links

| tavolo's API is straightforward and adopting its modules is as easy as it gets.
| In tavolo, you'll find implementations for basic layers like ``LayerNorm`` to complex modules like the Transformer's
  ``MultiHeadedSelfAttention``.
| For example, if we wanted to add head a multi-headed attention mechanism into our model, it would look something like:
|

.. code-block:: python3

    import tensorflow as tf
    import tavolo as tvl

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len),
        tvl.seq2seq.MultiHeadedSelfAttention(n_heads=8),  # <--- Add self attention
        tf.keras.layers.LSTM(n_lstm_units, return_sequences=True),
        tf.keras.layers.Dense(n_hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

.. _`tavolo`: https://github.com/eliorc/tavolo
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Kaggle`: https://www.kaggle.com
.. _`tf.keras`: https://www.tensorflow.org/guide/keras
