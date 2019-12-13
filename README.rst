.. raw:: html

    <div align="center">
      <img src="docs/source/_static/logo.png"><br><br>
    </div>

------------

.. image:: https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg
    :alt: Supported Python versions

.. image:: https://img.shields.io/badge/tensorflow-2.0-orange.svg
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


Documentation
-------------

| `Tavolo documentation`_

.. _`Tavolo documentation`: https://tavolo.readthedocs.io/

Showcase
--------

| tavolo's API is straightforward and adopting its modules is as easy as it gets.
| In tavolo, you'll find implementations for basic layers like `PositionalEncoding`_ to complex modules like the Transformer's
  `MultiHeadedAttention`_. You'll also find non-layer implementations that can ease development, like the `LearningRateFinder`_.
| For example, if we wanted to add head a Yang-style attention mechanism into our model and look for the optimal learning rate, it would look something like:

.. code-block:: python3

    import tensorflow as tf
    import tavolo as tvl

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len),
        tvl.seq2vec.YangAttention(n_units=64),  # <--- Add Yang style attention
        tf.keras.layers.Dense(n_hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.BinaryCrossentropy())

    # Run learning rate range test
    lr_finder = tvl.learning.LearningRateFinder(model=model)

    learning_rates, losses = lr_finder.scan(train_data, train_labels, min_lr=0.0001, max_lr=1.0, batch_size=128)

    ### Plot the results to choose your learning rate

.. _`tavolo`: https://github.com/eliorc/tavolo
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Kaggle`: https://www.kaggle.com
.. _`tf.keras`: https://www.tensorflow.org/guide/keras
.. _`PositionalEncoding`: https://tavolo.readthedocs.io/en/latest/embeddings.html#module-embeddings.PositionalEncoding
.. _`MultiHeadedAttention`: https://tavolo.readthedocs.io/en/latest/seq2seq.html#multi-headed-self-attention
.. _`LearningRateFinder`: https://tavolo.readthedocs.io/en/latest/learning.html#learning-rate-finder


Contributing
------------

| Want to contribute? Please read our `Contributing guide`_.

.. _`Contributing guide`: https://tavolo.readthedocs.io/en/latest/contributing.html
