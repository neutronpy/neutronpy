#!/usr/bin/env python
"""NeutronPy: Neutron scattering tools for scientific data analysis in python

NeutronPy is a collection of commonly used tools aimed at facilitating the
analysis of neutron scattering data. NeutronPy is built primarily using the
numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab)
routines for Instrument resolution calculations.

"""

import subprocess
from setuptools import setup

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
    v = subprocess.check_output(["git", "describe"]).rstrip().decode('ascii')
    if '-' in v:
        v = v.split('.')
        __version__ = '.'.join(v[:2]) + '.{0:d}'.format(int(v[2].split('-')[0]) + 1)
        __version__ += '.dev{0}'.format('.'.join(v[2].split('-')[1]))
    else:
        __version__ = v

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
                    setup_requires=['pytest-runner'],
                    tests_require=['pytest'],
                    ext_package='neutronpy',
                    package_data={'neutronpy': ['database/*.json', 'ui/*.ui']},
                    packages=['neutronpy', 'neutronpy.crystal', 'neutronpy.data', 'neutronpy.fileio',
                              'neutronpy.fileio.loaders', 'neutronpy.instrument', 'neutronpy.scattering',
                              'neutronpy.lsfit'],
                    entry_points={"console_scripts": ["neutronpy=neutronpy.gui:launch"]}, )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
