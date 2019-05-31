import os
import sys

from setuptools import setup
from setuptools.command.install import install

VERSION = '0.1.0'


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(name='tavolo',
      version=VERSION,
      description='Collection of deep learning modules and layers for the TensorFlow framework',
      url='https://github.com/eliorc/tavolo',
      author='Elior Cohen',
      license='MIT',
      packages=['tavolo'],
      install_requires=[
          'numpy',
          'tensorflow>=2.0.0-alpha0'],
      python_requires='>=3',
      cmdclass={
          'verify': VerifyVersionCommand})
