#!/usr/bin/env python

__version__ = '0.1alpha'

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import sys
import os

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
      version=__version__,
      description='Neutron Scattering Tools Python Library',
      author='David M Fobes',
      author_email='pseudocubic@gmail.com',
      url='https://github.com/pseudocubic/neutronpy',
#       install_requires=['numpy>=1.8.1', 'matplotlib>=1.3.1', 'scipy>=0.13.0'],
#       test_suite='neutronpy.tests.test_all',
      cmdclass={'build_ext': build_ext},
      ext_package='neutronpy',
      ext_modules=[Extension("fitting.kmpfit", ["src/kmpfit.pyx", "src/mpfit.c"], include_dirs=include_dirs)],
      package_data={'neutronpy': ['database/*.json']},
      packages=['neutronpy', 'neutronpy.fitting', 'neutronpy.resolution', 'neutronpy.tools', 'neutronpy.form_facs'],
)
