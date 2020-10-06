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
| In tavolo, you'll find implementations for layers like :ref:`positional_encoding` and non-layer implementations
  that can ease development, like the :ref:`learning_rate_finder`.
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


| You are welcome continue to the :doc:`install` page, or explore the different modules available:

.. toctree::
   :hidden:
   :maxdepth: 1

   install
   contributing


.. toctree::
   :maxdepth: 2
   :caption: Modules

   embeddings
   learning
   seq2vec


.. _`tavolo`: https://github.com/eliorc/tavolo
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Kaggle`: https://www.kaggle.com
.. _`tf.keras`: https://www.tensorflow.org/guide/keras


Contributing
------------

| Want to contribute? Please read our :doc:`contributing`.
