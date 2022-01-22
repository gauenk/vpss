#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""Patch Similarity-Search Package for Python."""
kwds = 'similarity search, patch matching, nearest neighbors, non-local image denosing'
setup(
    name='pss',
    version='1.0.0',
    description='A python implementation of HIDS',
    long_description=long_description,
    url='https://github.com/gauenk/pss',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords=kwds,
    install_requires=['numpy','torch','flake8'],
    packages=find_packages(),
)
