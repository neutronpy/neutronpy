#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from kapteyn import __version__ as version
from glob import glob
import sys, os

setup(name='neutronpy',
      version='0.1',
      description='Neutron Scattering Tools Python Library',
      author='David M Fobes',
      author_email='pseudocubic@gmail.com',
      url='https://github.com/pseudocubic/neutronpy',
      install_requires=['numpy>=1.8.1', 'six>=1.6.1'],
      packages=find_packages(),
      test_suite='neutronpy.tests.test_all',
      ext_modules=[
      Extension(
         "kmpfit",
         ["src/kmpfit.c", "src/mpfit.c"],
         include_dirs=include_dirs
      )
   ],
)
