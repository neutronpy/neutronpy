#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from Cython.Distutils import build_ext
from glob import glob
import sys, os

try:
   import numpy
except:
   sys.exit(1)

include_dirs = []
numdir = os.path.dirname(numpy.__file__)
ipath = os.path.join(numdir, numpy.get_include())
include_dirs.append(ipath)
include_dirs.append('src')

setup(name='neutronpy',
      version='0.1',
      description='Neutron Scattering Tools Python Library',
      author='David M Fobes',
      author_email='pseudocubic@gmail.com',
      url='https://github.com/pseudocubic/neutronpy',
      install_requires=['numpy>=1.8.1', 'six>=1.6.1'],
      packages=find_packages(),
      test_suite='neutronpy.tests.test_all',
      cmdclass = {'build_ext': build_ext},
      ext_modules=[
      Extension(
         "kmpfit",
         ["src/kmpfit.pyx", "src/mpfit.c"],
         include_dirs=include_dirs
      )
   ],
)
