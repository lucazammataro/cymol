# Module: cymol_3DLJP_02_1_3
# Title: Cymol Lennard Jones Potential 3D version
# Author: Luca Zammataro, Copyright (c) 2024
# Compile: python setup_cymol_3DLJP_02_1_3.py build_ext --inplace
# Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cymol_3DLJP_02_1_3.pyx"),
    include_dirs=[np.get_include()]
)
