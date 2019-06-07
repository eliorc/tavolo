from typing import Optional, Callable

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
        if self.scale_scheme not in {'triangular', 'triangular2', 'exp_rage'}:
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
