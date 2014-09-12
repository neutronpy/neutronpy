#!/usr/bin/env python
'''NeutronPy: Neutron scattering tools for scientific data analysis in python

NeutronPy is a collection of commonly needed tools aimed at facilitating the
analysis of neutron scattering data. NeutronPy is built primarily using the
numpy and scipy python libraries, with minor contributions from the kapteyn
library (least-squares fitting based on the C-implementation of MPFIT), and
a rewrite of ResLib 3.4c (MatLab) routines for Instrument resolution
calculations.

'''

import os
import sys
import re

# from distutils.core import Extension
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

DOCLINES = __doc__.split("\n")


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    cmdclass = {}

    with open('neutronpy/__init__.py') as f:
        __version__ = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

    include_dirs = ['src']

    try:
        import numpy
        include_dirs.append(os.path.join(os.path.dirname(numpy.__file__), numpy.get_include()))
    except ImportError:
        raise

    try:
        from Cython.Distutils import build_ext
        kmpfit_loc = "src/kmpfit.pyx"
        cmdclass['build_ext'] = build_ext
    except ImportError:
        kmpfit_loc = "src/kmpfit.c"

    metadata = dict(name='neutronpy',
                    version=__version__,
                    description=DOCLINES[0],
                    long_description="\n".join(DOCLINES[2:]),
                    author='David M Fobes',
                    maintainer='davidfobes',
                    download_url='https://github.com/neutronpy/neutronpy/releases',
                    url='https://github.com/neutronpy/neutronpy',
                    license='MIT',
                    platforms=["Windows", "Linux", "Mac OS X", "Unix"],
                    install_requires=['numpy>=1.8.0', 'scipy>=0.13.0', 'matplotlib>=1.3.0', 'cython>=0.20.0'],
                    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
                    test_suite='nose.collector',
                    cmdclass=cmdclass,
                    ext_package='neutronpy',
                    ext_modules=[Extension("kmpfit", [kmpfit_loc, "src/mpfit.c"], include_dirs=include_dirs)],
                    package_data={'neutronpy': ['database/*.json']},
                    packages=['neutronpy'],)

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
