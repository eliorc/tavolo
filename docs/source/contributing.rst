Contributing
============

| First of all, thanks for considering contributing code to tavolo!
| Before contributing please open an issue in the `Github repository`_ explaining the module you wish to contribute.
| Assuming the module is accepted, it will be tagged so in the issue opened so you can start implementing to avoid wasting contributor's time for code that won't be accepted.
| tavolo is built to compliment the `tf.keras`_ module, make sure your contributions are focused at it.
| Once your suggested module is accepted, follow the guidelines in :ref:`code_and_documentation` and :ref:`testing`, and once completed you can open a pull request to the ``dev`` branch.

.. note::

    Do not create pull requests into the ``master`` branch. Pull requests should be made to the ``dev`` branch, from which changes will be merged into ``master``
    on releases.


.. _`code_and_documentation`:

Code and Documentation
----------------------

| tavolo is open source, viewing the source code of a module and understanding every step in its implementation should be easy and straightforward, so users can trust the module they wish to use.
| To fulfill this requirement, follow these guidelines:

  #. Comments - Even if the code is clear, use comments to explain steps (`step comment example`_).
  #. Variable verbosity - Use verbose variable names that imply the meaning of their content, e.g. use ``mask`` instead of ``m``.
  #. Clear tensor shapes - When applying operations on tensors, include the shape of the result in a comment. (`tensor shape example`_).
  #. Format - `reStructuredText`_ is the documentation format use, and specifically PEP 287 (PyCharm's default) for class methods.
     On class level docstring, make sure you always include the following sections:

    * Arguments - For the ``__init__`` arguments (`Arguments section example`_).
    * Examples - For examples (`Examples section example`_)
    * References - For sources (articles etc.) for further reading (`References section example`_).

    If you are contributing a ``tf.keras.layers.Layer`` subclass, also include:

    * Input Shape - Input shape accepted by the layer's ``call`` method (`Input Shape section example`_).
    * Output Shape - Output shape accepted by the layer's ``call`` method (`Output Shape section example`_).

.. _`Github repository`: https://github.com/eliorc/tavolo
.. _`tf.keras`: https://www.tensorflow.org/guide/keras
.. _`step comment example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L82
.. _`tensor shape example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L83
.. _`reStructuredText`: https://www.sphinx-doc.org/en/stable/usage/restructuredtext/index.html
.. _`PEP 287`: https://www.python.org/dev/peps/pep-0287/
.. _`Arguments section example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L9
.. _`Examples section example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L29
.. _`References section example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L41
.. _`Input Shape section example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L16
.. _`Output Shape section example`: https://gist.github.com/eliorc/7095070fb371a41eb3151d4cf73b25d2#file-layer_normalization-py-L23


.. _`testing`:

Testing
-------

| Tavolo uses `pytest`_ and `codecov`_ for its testing. Make sure you write your tests to cover the full functionality of the contributed code.
| The tests should be written as a separate file for each module and in the destination ``tests/<parent_module>/<module_name>_test.py``.
| For example for the module ``tvl.normalization.LayerNormalization``, the tests should be written in ``tests/normalization/layer_normalization_test.py``.
| It is quite difficult to define in advance which tests are mandatory, but you can draw insipration from the existing modules.
| In the specific case of ``tf.keras.layers.Layer`` implementation, always include:

  #. ``test_shapes()`` - Given accepted input shapes, make sure the output shape is as expected (`test_shapes() example`_).
  #. ``test_masking()`` - Make sure layer supports masking (`test_masking() example`_).
  #. ``test_serialization()`` - Make sure layer can be saved and loaded using ``get_config`` and ``from_config`` (`test_serialization() example`_).

| If possible, also include ``test_logic()`` for evaluating expected output given known input (`test_logic() example`_).
|
| When done, run tests locally to make sure everything works fine, to do so, make sure you have installed the test requirements from the `requirements/test.py`_ file and run tests locally using the following command from the main directory

.. code-block:: bash

    pytest --cov=tavolo tests/

| Strive for 100% coverage, and if all is well, create a pull request (to the ``dev`` branch) and it will be added to the package in a following release.

.. _`pytest`: https://docs.pytest.org/en/latest/
.. _`codecov`: https://codecov.io/
.. _`test_shapes() example`: https://gist.github.com/eliorc/6ac98485b0606045f2412a587165176a#file-layer_normalization_test-py-L6
.. _`test_masking() example`: https://gist.github.com/eliorc/6ac98485b0606045f2412a587165176a#file-layer_normalization_test-py-L26
.. _`test_serialization() example`: https://gist.github.com/eliorc/6ac98485b0606045f2412a587165176a#file-layer_normalization_test-py-L57
.. _`test_logic() example`: https://gist.github.com/eliorc/6ac98485b0606045f2412a587165176a#file-layer_normalization_test-py-L44
.. _`requirements/test.py`: https://github.com/eliorc/tavolo/blob/master/requirements/test.txt
