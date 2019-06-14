import tempfile
import math
from typing import Optional, Callable, Tuple, List
from contextlib import suppress

import tensorflow as tf
import numpy as np


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


class LearningRateFinder:
    """
    Learning rate finding utility for conducting the "LR range test", see article reference for more information

    Use the ``scan`` method for finding the loss values for learning rates in the given range

    Arguments
    ---------

    - `model` (``tf.keras.Model``): Model for conduct test for. Must call ``model.compile`` before using this utility

    Examples
    --------

    Run an learning rate range test in the domain ``[0.0001, 1.0]``

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
