'''
Created on May 19, 2014

@author: davidfobes
'''
import json
import os
import numpy as np

with open(os.path.join(os.path.dirname(__file__), 
                       "database/mag_ion_j.json"), 'r') as infile:
    magIonJ = json.load(infile)
    
with open(os.path.join(os.path.dirname(__file__), 
                       "database/periodic_table.json"), 'r') as infile:
    periodicTable = json.load(infile)
    
with open(os.path.join(os.path.dirname(__file__), 
                       "database/scat_len.json"), 'r') as infile:
    scatLen = json.load(infile)

h = 6.626e-34  # Units: J * s
hbar = h / (2 * np.pi)  # Units: J * s
m_neutron = 1.674927351e-27  # Units: kg
kB = 0.086173323  # Units: meV / K
J2meV = 1. / 1.602176565e-19 * 1.e3
m2A = 1.e10

class Neutron():
    def __init__(self, e=None, l=None, v=None, k=None, temp=None, freq=None):

        if e is None:
            print('no energy')
            if l is not None:
                self.e = h ** 2 / (2. * m_neutron * (l / m2A) ** 2) * J2meV
            elif v is not None:
                self.e = 1. / 2. * m_neutron * v ** 2 * J2meV
            elif k is not None:
                self.e = h ** 2 / (2. * m_neutron * ((2. * np.pi / k) / m2A) ** 2) * J2meV
            elif temp is not None:
                self.e = kB * temp
            elif freq is not None:
                self.e = hbar * freq * 2. * np.pi * J2meV

        self.l = np.sqrt(h ** 2 / (2. * m_neutron * self.e / J2meV)) * m2A
        self.v = np.sqrt(2. * self.e / J2meV / m_neutron)
        self.k = 2. * np.pi / self.l
        self.temp = self.e / kB
        self.freq = self.e / J2meV / hbar / 2. / np.pi / 1.e12
    
    
    def printValues(self):
        output = '''
        Energy: {0:3.3f} meV
        Wavelength: {1:3.3f} A
        Wavevector: {2:3.3f} A^-1
        Velocity: {3:3.3f} m/s
        Temperature: {4:3.3f} K
        Frequency: {5:3.3f} THz
        '''.format(self.e, self.l, self.k, self.v, self.temp, self.freq)

        print(output)