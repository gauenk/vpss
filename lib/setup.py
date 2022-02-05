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
long_description="""Video Patch Similarity-Search Package for Python. This is commonly used in non-local methods for image denoising."""
kwds = 'similarity search, patch matching, nearest neighbors, non-local image denosing, vnlb, numba, gpu, cuda'
setup(
    name='vpss',
    version='100.100.100',
    description='A Python implementation of Video Patch Similarity Search',
    long_description=long_description,
    url='https://github.com/gauenk/vpss',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords=kwds,
    install_requires=['numpy','numba','torch','flake8'],
    packages=find_packages(),
)
