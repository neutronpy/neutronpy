#!/usr/bin/env python
"""NeutronPy: Neutron scattering tools for scientific data analysis in python

NeutronPy is a collection of commonly used tools aimed at facilitating the
analysis of neutron scattering data. NeutronPy is built primarily using the
numpy and scipy python libraries, with a translation of ResLib 3.4c (MatLab)
routines for Instrument resolution calculations.

"""

import os
import re
import subprocess
import warnings
from math import ceil, log10

from setuptools import setup

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Natural Language :: English
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS :: MacOS X
"""

DOCLINES = __doc__.split("\n")


def get_version():
    r"""Determines version of package using either git describe or via the
        folder name. Defaults to 0.0.0 if none is found, and warns user to
        use a supported install method.
    """
    vpat = re.compile(r"^([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?$")
    try:
        v = subprocess.check_output(["git", "describe", "--tags"]).rstrip().decode('ascii')
    except subprocess.CalledProcessError:
        ospat = re.compile(r".*neutronpy-(.+)")
        osmatch = ospat.match(os.path.dirname(os.path.realpath(__file__)))
        if osmatch is not None:
            v = osmatch.groups()[0]
        else:
            warnings.warn("Cannot find current version of neutronpy, please use supported install method.")
            v = "0.0.0"

    if '-' in v:
        v, ntag = v.split('-')[0:2]
        vmatch = vpat.match(v)

        epoch, major, minor, patch, pre, pretype, prever, post, postver, dev, devver = vmatch.groups()

        if post is not None:
            post = "post{0}".format(int(postver + 1))
        elif pre is not None:
            pre = "{0}{1}".format(pretype, int(prever) + 1)
        else:
            patch = "{0}".format(int(patch) + 1)

        if dev is not None:
            devver = int(devver)
            ntag_mag = ceil(log10(ntag))
            dev = "dev{0}".format(int(devver) * 10 ** ntag_mag + ntag)
        else:
            dev = "dev{0}".format(ntag)

        front_vers = [major, minor, patch]
        back_vers = [pre, post, dev]

        __version__ = '.'.join([item.strip('.') for item in front_vers if item is not None])
        __version__ += '.'.join([item.strip('.') for item in back_vers if item is not None])

        if epoch is not None:
            __version__ = "{0}!{1}".format(epoch, __version__)
    else:
        __version__ = v

    return __version__


def setup_package():
    r"""Setup package function
    """

    metadata = dict(name='neutronpy',
                    version=get_version(),
                    description=DOCLINES[0],
                    long_description="\n".join(DOCLINES[2:]),
                    author='David M Fobes',
                    author_email='dfobes@lanl.gov',
                    maintainer='davidfobes',
                    download_url='https://github.com/neutronpy/neutronpy/releases',
                    url='https://github.com/neutronpy/neutronpy',
                    license='MIT',
                    platforms=["Windows", "Linux", "Mac OS X", "Unix"],
                    install_requires=['numpy>=1.10', 'scipy>=1.0', 'matplotlib>=2.0', 'lmfit>=0.9.5', 'h5py'],
                    setup_requires=['pytest-runner'],
                    tests_require=['pytest','mock', 'codecov'],
                    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
                    ext_package='neutronpy',
                    package_data={'neutronpy': ['database/*.json', 'ui/*.ui']},
                    packages=['neutronpy', 'neutronpy.crystal', 'neutronpy.data', 'neutronpy.fileio',
                              'neutronpy.fileio.loaders', 'neutronpy.instrument', 'neutronpy.scattering',
                              'neutronpy.lsfit'],
                    entry_points={"console_scripts": ["neutronpy=neutronpy.gui:launch"]}, )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
