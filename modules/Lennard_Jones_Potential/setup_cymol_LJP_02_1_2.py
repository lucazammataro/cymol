'''
Title: Cymol Lennard Jones Potential 2D version
Author: Luca Zammataro, Copyright (c) 2024
Code: setup
Compile: python setup_cymol_LJP_02_1_2.py build_ext --inplace
Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cymol_LJP_02_1_2.pyx"),
    include_dirs=[np.get_include()]
)
