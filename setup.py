#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='neutronpy',
      version='0.1',
      description='Neutron Scattering Tools Python Library',
      author='David M Fobes',
      author_email='pseudocubic@gmail.com',
      url='https://github.com/pseudocubic/neutronpy',
      install_requires=['numpy>=1.8.1', 'matplotlib>=1.3.1', 'scipy>=0.14', 'six>=1.6.1'],
      packages=find_packages(),
      test_suite='neutronpy.tests.test_all',
      )
