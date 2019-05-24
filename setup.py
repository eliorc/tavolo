from setuptools import setup

setup(name='tavolo',
      version='0.1.0',
      description='Collection of deep learning modules and layers for the TensorFlow framework',
      url='https://github.com/eliorc/tavolo',
      author='Elior Cohen',
      license='MIT',
      packages=['tavolo'],
      install_requires=[
          'numpy',
          'tensorflow>=2.0.0-alpha0'])
