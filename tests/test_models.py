r'''Tests special models

'''
import unittest
from neutronpy import models


class ModelTests(unittest.TestCase):
    '''Unit testing for special Models
    '''
    def test_harmonic_oscillator(self):
        '''Test harmonic oscillator
        '''
        models.harmonic_oscillator()

    def test_acoustic_phonon(self):
        '''Test acoustic phonon dispersion
        '''
        models.acoustic_phonon_dispersion()

    def test_optical_phonon(self):
        '''Test optical phonon dispersion
        '''
        models.optical_phonon_disperions()


if __name__ == '__main__':
    unittest.main()
