# -*- coding: utf-8 -*-
'''Testing for Energy conversions

'''
import unittest
import numpy as np
from neutronpy.energy import Energy


class EnergyTest(unittest.TestCase):
    '''Unit tests for Energy conversion
    '''
    def test_energy(self):
        '''Test that the output string is correct
        '''
        energy = Energy(energy=25.)
        self.assertEqual(np.round(energy.energy), 25.0)
        self.assertEqual(np.round(energy.wavelength, 4), 1.8089)
        self.assertEqual(np.round(energy.wavevector, 3), 3.473)
        self.assertEqual(np.round(energy.velocity, 3), 2186.967)
        self.assertEqual(np.round(energy.temperature, 3), 290.113)
        self.assertEqual(np.round(energy.frequency, 3), 6.045)

        stringtest = u"\nEnergy: {0:3.3f} meV"
        stringtest += u"\nWavelength: {1:3.3f} Å"
        stringtest += u"\nWavevector: {2:3.3f} 1/Å"
        stringtest += u"\nVelocity: {3:3.3f} m/s"
        stringtest += u"\nTemperature: {4:3.3f} K"
        stringtest += u"\nFrequency: {5:3.3f} THz\n"
        stringtest = stringtest.format(25.0, 1.8089, 3.473, 2186.967,
                                       290.113, 6.045)

        self.assertTrue(energy.values == stringtest)

    def test_energy_setters(self):
        '''Tests the energy setters
        '''
        e = Energy(energy=25.)

        e.energy = 25
        self.assertEqual(np.round(e.wavelength, 4), 1.8089)

        e.wavevector = 3.5
        self.assertEqual(np.round(e.energy, 1), 25.4)

        e.velocity = 2180
        self.assertEqual(np.round(e.energy, 1), 24.8)

        e.temperature = 290
        self.assertEqual(np.round(e.energy, 1), 25.0)

        e.frequency = 6
        self.assertEqual(np.round(e.energy, 1), 24.8)

        e.wavelength = 1.9
        self.assertEqual(np.round(e.energy, 1), 22.7)


if __name__ == "__main__":
    unittest.main()