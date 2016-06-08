#!/usr/bin/env python
"""NeutronPy: Neutron scattering tools for scientific data analysis in python

NeutronPy is a collection of commonly used tools aimed at facilitating the
analysis of neutron scattering data. NeutronPy is built primarily using the
numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab)
routines for Instrument resolution calculations.

"""

import os
import sys
import re

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

DOCLINES = __doc__.split("\n")


def setup_package():
    r"""Setup package function
    """
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    with open('neutronpy/__init__.py') as f:
        __version__ = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

    include_dirs = []

    try:
        import numpy
        include_dirs.append(os.path.join(os.path.dirname(numpy.__file__), numpy.get_include()))
    except ImportError:
        raise

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
                    install_requires=['numpy', 'scipy', 'matplotlib', 'lmfit'],
                    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
                    test_suite='nose.collector',
                    ext_package='neutronpy',
                    package_data={'neutronpy': ['database/*.json', 'ui/*.ui']},
                    packages=['neutronpy', 'neutronpy.crystal', 'neutronpy.data', 'neutronpy.fileio',
                              'neutronpy.fileio.loaders', 'neutronpy.instrument', 'neutronpy.scattering',
                              'neutronpy.lsfit'],
                    entry_points={"console_scripts": ["neutronpy=neutronpy.gui:launch"]}, )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
