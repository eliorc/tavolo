.. raw:: html

    <div align="center">
      <img src="docs/source/_static/logo.png"><br><br>
    </div>

------------

.. image:: https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7-blue.svg
    :alt: Supported Python versions

.. image:: https://img.shields.io/badge/tensorflow-2.0.0--alpha0-orange.svg
    :alt: Supported TensorFlow versions

.. image:: https://codecov.io/gh/eliorc/tavolo/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/eliorc/tavolo
    :alt: Code test coverage

.. image:: https://circleci.com/gh/eliorc/tavolo.svg?style=svg
    :target: https://circleci.com/gh/eliorc/tavolo
    :alt: CircleCI status

Tavolo
======

| `tavolo`_ aims to package together valuable modules and functionality written for `TensorFlow`_ high-level Keras API for ease of use.
| You see, the deep learning world is moving fast, and new ideas keep on coming.
| tavolo gathers implementations of these useful ideas from the community (by contribution, from `Kaggle`_ etc.)
  and makes them accessible in a single PyPI hosted package that compliments the `tf.keras`_ module.
|
| *Notice: tavolo is developed for TensorFlow 2.0 (right now on alpha), most modules will work with earlier versions but some won't (like LayerNorm)*

Documentation
-------------

| `Tavolo documentation`_

.. _`Tavolo documentation`: https://tavolo.readthedocs.io/

Showcase
--------

| tavolo's API is straightforward and adopting its modules is as easy as it gets.
| In tavolo, you'll find implementations for basic layers like `LayerNorm`_ to complex modules like the Transformer's
  `MultiHeadedSelfAttention`_.
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
.. _`LayerNorm`: https://tavolo.readthedocs.io/en/latest/normalization.html#layer-norm
.. _`MultiHeadedSelfAttention`: https://tavolo.readthedocs.io/en/latest/seq2seq.html#multi-headed-self-attention
