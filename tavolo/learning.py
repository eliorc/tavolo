import math
import tempfile
from contextlib import suppress
from typing import Optional, Callable, Tuple, List, Any

import numpy as np
import tensorflow as tf


class CyclicLearningRateCallback(tf.keras.callbacks.Callback):
    """
    Apply cyclic learning rate. Supports the following scale schemes:

    - ``triangular`` - Triangular cycle
    - ``triangular2`` - Triangular cycle that shrinks amplitude by half each cycle
    - ``exp_range`` - Triangular cycle that shrinks amplitude by ``gamma ** <cycle iterations>`` each cycle


    Arguments
    ---------

    - `base_lr` (``float``): Lower boundary of each cycle
    - `max_lr` (``float``): Upper boundary of each cycle, may not be reached depending on the scaling function
    - `step_size` (``int``): Number of batches per half-cycle (step)
    - `scale_scheme` (``str``): One of ``{'triangular', 'triangular2', 'exp_range'}``. If ``scale_fn`` is passed, this argument is ignored
    - `gamma` (``float``): Constant used for the ``exp_range``'s ``scale_fn``, used as (``gamma ** <cycle iterations>``)
    - `scale_fn` (``callable``): Custom scaling policy, accepts cycle index / iterations depending on the ``scale_mode`` and must return a value in the range [0, 1]. If passed, ignores ``scale_scheme``
    - `scale_mode` (``str``): Define whether ``scale_fn`` is evaluated on cycle index or cycle iterations


    Examples
    --------

    Apply a triangular cyclic learning rate (default), with a step size of 2000 batches

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        clr = tvl.learning.CyclicLearningRateCallback(base_lr=0.001, max_lr=0.006, step_size=2000)

        model.fit(X_train, Y_train, callbacks=[clr])

    Apply a cyclic learning rate that shrinks amplitude by half each cycle

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        clr = tvl.learning.CyclicLearningRateCallback(base_lr=0.001, max_lr=0.006, step_size=2000, scale_scheme='triangular2')

        model.fit(X_train, Y_train, callbacks=[clr])

    Apply a cyclic learning rate with a custom scaling function

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        scale_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2))
        clr = tvl.learning.CyclicLearningRateCallback(base_lr=0.001, max_lr=0.006, step_size=2000, scale_fn=scale_fn)

        model.fit(X_train, Y_train, callbacks=[clr])


    References
    ----------

    - `Cyclical Learning Rates for Training Neural Networks`_
    - `Original implementation`_

    .. _`Cyclical Learning Rates for Training Neural Networks`: https://arxiv.org/abs/1506.01186
    .. _`Original implementation`: https://github.com/bckenstler/CLR

    """

    def __init__(self, base_lr: float = 0.001,
                 max_lr: float = 0.006,
                 step_size: int = 2000.,
                 scale_scheme: str = 'triangular',
                 gamma: float = 1.,
                 scale_fn: Optional[Callable[[int], float]] = None,
                 scale_mode: str = 'cycle'):
        """

        :param base_lr: Lower boundary of each cycle
        :param max_lr: Upper boundary of each cycle, may not be reached depending on the scaling function
        :param step_size: Number of batches per half-cycle (step)
        :param scale_scheme: One of ``{'triangular', 'triangular2', 'exp_range'}``. If ``scale_fn`` is passed,
            this argument is ignored.
        :param gamma: Constant used for the ``exp_range``'s ``scale_fn``, used as (``gamma ** <cycle iterations>``)
        :param scale_fn: Custom scaling policy, accepts cycle index / iterations depending on the ``scale_mode``
            and must return a value in the range [0, 1]. If passed, ignores ``scale_scheme``
        :param scale_mode: Define whether ``scale_fn`` is evaluated on cycle index or cycle iterations
        """

        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.scale_scheme = scale_scheme
        self.gamma = gamma

        if not scale_fn:
            self._set_scale_scheme()
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        # Init
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = dict()

    def clr(self) -> float:
        """
        Calculate learning rate

        :return: Learning rate value
        """

        # Calculate learning rate
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        clr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * \
              self.scale_fn(cycle if self.scale_mode == 'cycle' else self.clr_iterations)

        return clr

    def on_train_begin(self, logs: Optional[dict] = None):

        logs = logs or dict()

        # Apply learning rate to optimizer
        tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr if self.clr_iterations == 0 else self.clr())

    def on_train_batch_end(self, batch: tf.Tensor, logs: Optional[dict] = None):

        logs = logs or dict()

        # Step
        self.trn_iterations += 1
        self.clr_iterations += 1

        # Log history
        self.history.setdefault('lr', list()).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', list()).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, list()).append(v)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def _set_scale_scheme(self):
        """
        Applies scale mode and function for supported (non-custom) scaling schemes

        """

        # Check for supported scale schemes
        if self.scale_scheme not in {'triangular', 'triangular2', 'exp_range'}:
            raise ValueError('{} is not a supported scale scheme'.format(self.scale_scheme))

        # Set scheme
        if self.scale_scheme == 'triangular':
            self.scale_fn = lambda x: 1
            self.scale_mode = 'cycle'
        elif self.scale_scheme == 'triangular2':
            self.scale_fn = lambda x: 1 / (2 ** (x - 1))
            self.scale_mode = 'cycle'
        elif self.scale_scheme == 'exp_range':
            self.scale_fn = lambda x: self.gamma ** x
            self.scale_mode = 'iterations'


class DataMapCallback(tf.keras.callbacks.Callback):
    """
    Gather training dynamics for data map generation. Assumes a binary or multi-class model, no support for multi label.

    Arguments
    ---------

    - `dataset` (``tf.data.: Dataset``): Usually, as the paper suggests, this is the training dataset. It should be:

        - Non-shuffled, so each iteration over the dataset should yield samples in the same order
        - Already batched, the ``.batch(n)`` method should already be applied on this dataset
        - Should yield batches of ``(features, labels)``, sample weights are not supported

    - `outputs_to_probabilities` (``Optional[Callable[[Any], tf.Tensor]]``): Callable to convert model's output to
        probabilities. Use this if the model outputs logits, dictionary or any other form which is not a tensor
        of probabilities. Defaults to ``None``.
    - `sparse_labels` (``bool``): Set to ``True`` if the labels are given as integers (not one hot encoded). Defaults
        to ``False``.

    Attributes
    ----------

    - `gold_labels_probabilities` (``np.ndarray``): Gold label predicted probabilities. With the shape of
        ``(n_samples, n_epochs)`` and ``(i, j)`` is the probability of the gold label for sample ``i`` at epoch ``j``.
    - `confidence` (``np.ndarray``): Mean of true label probability across epochs.
    - `variability` (``np.ndarray``): Standard deviation of true label probability across epochs.
    - `correctness` (``np.ndarray``): Fraction of times correctly predicted across epochs


    Examples
    --------

    Calculate training dynamics during training

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        # Load dataset
        train = ... # Instance of dataset
        train_unshuffled = ... # Instance of dataset, unshuffled so that each iteration over the dataset would yield
                               # samples in the same order

        # Prepare
        train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        train = train_unshuffled.batch(BATCH_SIZE * 10)  # No gradient updates in data map, can use bigger batches

        # Create the datamap callback
        datamap = tvl.learning.DatMaCallback(dataset=train_unshuffled)

        # Train
        model.fit(train, epochs=N_EPOCHS, callbacks=[datamap])

        # Get training dynamics
        confidence, variability, correctness = datamap.confidence, datamap.variability, datamap.correctness


    Calculate training dynamics from a model that outputs logits (and NOT probabilities)

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        # Create the datamap callback - using the outputs_to_predictions option
        datamap = tvl.learning.DatMaCallback(dataset=train_unshuffled, outputs_to_probabilities=tf.nn.softmax)

        # Train
        model.fit(train, epochs=N_EPOCHS, callbacks=[datamap])

    References
    ----------

    - `Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics`_

    .. _`Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics`: https://arxiv.org/pdf/2009.10795

    """

    # TODO - The implementation saves all the gold label probabilities across epochs for the training dynamics
    #        computations. This can be optimized by calculating a running version of each training dynamic.
    #        Once tfp.stats releases RunningVariance and RunningMean to the stable tfp versions - training dynamics
    #        calculations should be reimplemented doing this, thus avoiding (n_epochs - 1) * n_samples floating points
    #        memory usage.

    def __init__(self, dataset: tf.data.Dataset,
                 outputs_to_probabilities: Optional[Callable[[Any], tf.Tensor]] = None,
                 sparse_labels: bool = False):
        """

        :param dataset: Dataset. Usually, as the paper suggests, this is the training dataset. It should be:

             - Non-shuffled, so each iteration over the dataset should yield samples in the same order
             - Already batched, the ``.batch(n)`` method should already be applied on this dataset
             - Should yield batches of ``(features, labels)``, sample weights are not supported

        :param outputs_to_probabilities: Callable to convert model's output to probabilities. Use this if the model
            outputs logits, dictionary or any other form which is not a vector of probabilities.
        :param sparse_labels: Set to ``True`` if the labels are given as integers (not one hot encoded)
        """

        self._dataset = dataset
        self._outputs2probabilities = outputs_to_probabilities
        self._sparse_labels = sparse_labels

        # Stores the probabilities for the gold labels after each epoch,
        # e.g. self._gold_labels_probabilities[i] == gold label probabilities at the end of epoch i
        self._gold_labels_probabilities = None

    def on_epoch_end(self, epoch, logs=None):

        # Gather gold label probabilities over the dataset
        gold_label_probabilities = list()
        for x, y in self._dataset:
            probabilities = self.model.predict(x)

            # Convert outputs to probabilities if necessary
            if self._outputs2probabilities is not None:
                probabilities = self._outputs2probabilities(probabilities)

            # Convert to probabilities if labels are sparse, e.g. [1, 2] -> [[0, 1, 0], [0, 0, 1]]
            if self._sparse_labels:
                y = tf.one_hot(y, depth=probabilities.shape[-1])

            # Extract the gold label probabilities from the predictions
            if tf.rank(tf.squeeze(y)) == 1:  # Labels binary e.g. y = [1, 0, 0, 1...], len(y) == batch_size
                # Squeeze to remove redundant dimensions
                probabilities, y = tf.squeeze(probabilities), tf.squeeze(y)

                batch_gold_label_probabilities = tf.where(y == 0, 1 - probabilities, probabilities)
            elif tf.rank(tf.squeeze(y)) == 2:  # One hot labels e.g. y = [[1,0,0], [0,1,0] ...], len(y == batch_size)
                # Verify labels are NOT multi-labels (e.g. [[1,0,1], [1,1,0], ...]
                if not tf.reduce_all(tf.reduce_sum(tf.cast(y == 1, tf.int32), axis=-1) == 1):
                    raise ValueError('DataMapCallback does not support multi-label classification')

                batch_gold_label_probabilities = tf.boolean_mask(probabilities, tf.cast(y, tf.bool)).numpy()
            else:
                raise ValueError(
                    'tf.squeeze(y) (y == labels from the dataset) must be of rank 1 for binary classification or '
                    '2 for multi class. Instead got ({})'.format(tf.rank(tf.squeeze(y))))

            # Gather gold label probabilities
            gold_label_probabilities = np.append(gold_label_probabilities, [batch_gold_label_probabilities])

        # Gather probabilities
        if self._gold_labels_probabilities is None:  # Happens only on first iteration
            self._gold_labels_probabilities = np.expand_dims(gold_label_probabilities, axis=-1)
        else:
            stack = [self._gold_labels_probabilities, np.expand_dims(gold_label_probabilities, axis=-1)]
            self._gold_labels_probabilities = np.hstack(stack)

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Gold label predicted probabilities. With the shape of ``(n_samples, n_epochs)`` and ``(i, j)`` is the
        probability of the gold label for sample ``i`` at epoch ``j``

        :return: Gold label probabilities
        """

        return self._gold_labels_probabilities

    @property
    def confidence(self) -> np.ndarray:
        """
        Mean of true label probability across epochs

        :return: Confidence
        """
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Standard deviation of true label probability across epochs

        :return: Variability
        """

        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Fraction of times correctly predicted across epochs

        :return: Correctness
        """
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)


class LearningRateFinder:
    """
    Learning rate finding utility for conducting the "LR range test", see article reference for more information

    Use the ``scan`` method for finding the loss values for learning rates in the given range

    Arguments
    ---------

    - `model` (``tf.keras.Model``): Model for conduct test for. Must call ``model.compile`` before using this utility

    Examples
    --------

    Run a learning rate range test in the domain ``[0.0001, 1.0]``

    .. code-block:: python3

        import tensorflow as tf
        import tavolo as tvl

        train_data = ...
        train_labels = ...

        # Build model
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(784,)),
                                     tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

        # Must call compile with optimizer before test
        model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.CategoricalCrossentropy())

        # Run learning rate range test
        lr_finder = tvl.learning.LearningRateFinder(model=model)

        learning_rates, losses = lr_finder.scan(train_data, train_labels, min_lr=0.0001, max_lr=1.0, batch_size=128)

        ### Plot the results to choose your learning rate


    References
    ----------
    - `Cyclical Learning Rates for Training Neural Networks`_

    .. _`Cyclical Learning Rates for Training Neural Networks`: https://arxiv.org/abs/1506.01186

    """

    def __init__(self, model: tf.keras.Model):
        """
        :param model: Model for conduct test for. Must call ``model.compile`` before using this utility
        """

        self._model = model
        self._lr_range = None
        self._iteration = None
        self._learning_rates = None
        self._losses = None

    def scan(self, x, y,
             min_lr: float = 0.0001,
             max_lr: float = 1.0,
             batch_size: Optional[int] = None,
             steps: int = 100) -> Tuple[List[float], List[float]]:
        """
        Scans the learning rate range ``[min_lr, max_lr]`` for loss values

        :param x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs)
          - A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs)
          - A dict mapping input names to the corresponding array/tensors, if the model has named inputs
          - A ``tf.data`` dataset or a dataset iterator. Should return a tuple of either ``(inputs, targets)`` or
          ``(inputs, targets, sample_weights)``
          - A generator or ``keras.utils.Sequence`` returning ``(inputs, targets)`` or ``(inputs, targets, sample weights)``
        :param y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with ``x`` (you cannot have Numpy inputs and
          tensor targets, or inversely). If ``x`` is a dataset, dataset
          iterator, generator, or ``tf.keras.utils.Sequence`` instance, ``y`` should
          not be specified (since targets will be obtained from ``x``).
        :param min_lr: Minimum learning rate
        :param max_lr: Maximum learning rate
        :param batch_size: Number of samples per gradient update.
          Do not specify the ``batch_size`` if your data is in the
          form of symbolic tensors, dataset, dataset iterators,
          generators, or ``tf.keras.utils.Sequence`` instances (since they generate batches)
        :param steps: Number of steps to scan between min_lr and max_lr
        :return: Learning rates, losses documented

        """

        # Prerequisites
        self._iteration = 0
        self._learning_rates = list()
        self._losses = list()

        # Save initial values
        initial_checkpoint = tempfile.NamedTemporaryFile()
        self._model.save_weights(initial_checkpoint.name)  # Save original weights
        initial_learning_rate = tf.keras.backend.get_value(self._model.optimizer.lr)  # Save original lr

        # Build range
        self._lr_range = np.linspace(start=min_lr, stop=max_lr, num=steps)

        # Scan
        tf.keras.backend.set_value(self._model.optimizer.lr, self._lr_range[self._iteration])
        scan_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: self._on_batch_end(batch, logs))

        self._model.fit(x, y, batch_size=batch_size, epochs=1, steps_per_epoch=steps,
                        verbose=0, callbacks=[scan_callback])

        # Restore initial values
        self._model.load_weights(initial_checkpoint.name)  # Restore original weights
        tf.keras.backend.set_value(self._model.optimizer.lr, initial_learning_rate)  # Restore original lr

        return self._learning_rates, self._losses

    def _on_batch_end(self, batch: tf.Tensor, logs: dict):
        # Save learning rate and corresponding loss
        self._learning_rates.append(
            tf.keras.backend.get_value(self._model.optimizer.lr))

        self._losses.append(
            logs['loss'])

        # Stop on exploding gradient
        if math.isnan(logs['loss']):
            self._model.stop_training = True
            return

        # Apply next learning rate
        with suppress(IndexError):
            tf.keras.backend.set_value(self._model.optimizer.lr, self._lr_range[self._iteration + 1])
            self._iteration += 1
