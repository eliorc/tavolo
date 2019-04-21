from setuptools import setup
import tavolo

setup(name=tavolo.__name__,
      version=tavolo.__version__,
      description='Collection of deep learning modules and layers for the TensorFlow framework',
      url='https://github.com/eliorc/tavolo',
      author='Elior Cohen',
      license='MIT',
      packages=['tavolo'],
      install_requires=[
          'numpy',
          'tensorflow'])
