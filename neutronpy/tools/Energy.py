'''
Created on May 22, 2014

@author: davidfobes
'''
from neutronpy.constants import joules2meV
import numpy as np
from scipy import constants


class Neutron():
    '''Class containing the most commonly used properties of a neutron beam
    given some initial input, e.g. energy, wavelength, wavevector,
    temperature, or frequency'''

    def __init__(self, e=None, l=None, v=None, k=None, temp=None, freq=None):
        if e is None:
            if l is not None:
                self.e = constants.h ** 2 / (2. * constants.m_n *
                                             (l / 1.e10) ** 2) * joules2meV
            elif v is not None:
                self.e = 1. / 2. * constants.m_n * v ** 2 * joules2meV
            elif k is not None:
                self.e = (constants.h ** 2 /
                          (2. * constants.m_n *
                           ((2. * np.pi / k) / 1.e10) ** 2) * joules2meV)
            elif temp is not None:
                self.e = constants.k * temp * joules2meV
            elif freq is not None:
                self.e = constants.hbar * freq * 2. * np.pi * \
                            joules2meV * 1.e12
        else:
            self.e = e

        self.l = np.sqrt(constants.h ** 2 / (2. * constants.m_n * self.e /
                                             joules2meV)) * 1.e10
        self.v = np.sqrt(2. * self.e / joules2meV / constants.m_n)
        self.k = 2. * np.pi / self.l
        self.temp = self.e / constants.k / joules2meV
        self.freq = self.e / joules2meV / constants.hbar / 2. / np.pi / 1.e12

    def printValues(self):
        print('''
Energy: {0:3.3f} meV
Wavelength: {1:3.3f} A
Wavevector: {2:3.3f} A^-1
Velocity: {3:3.3f} m/s
Temperature: {4:3.3f} K
Frequency: {5:3.3f} THz
'''.format(self.e, self.l, self.k, self.v, self.temp, self.freq))
