from setuptools import setup

VERSION = '0.7.0'

setup(name='tavolo',
      version=VERSION,
      description='Collection of deep learning modules and layers for the TensorFlow framework',
      url='https://github.com/eliorc/tavolo',
      author='Elior Cohen',
      classifiers=['License :: OSI Approved :: MIT License'],
      packages=['tavolo'],
      install_requires=[
          'numpy'],
      python_requires='>=3.5')
