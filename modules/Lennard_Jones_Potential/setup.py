'''
Module: cymol_LJP_02_1_1
Title: Cymol Lennard Jones Potential
Author: Luca Zammataro, Copyright (c) 2024
Compile: python setup.py build_ext --inplace
Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c
This project is licensed under the GNU General Public License v3.0
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cymol_LJP_02_1_1.pyx"),
    include_dirs=[np.get_include()]
)
