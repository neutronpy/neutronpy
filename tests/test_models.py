r'''Tests special models

'''
import unittest
import numpy as np
from neutronpy import models


class ModelTests(unittest.TestCase):
    '''Unit testing for special Models
    '''
    def test_harmonic_oscillator(self):
        '''Test harmonic oscillator
        '''
        ps = [0, 1, 0.5, 0]
        pd = [0, 1, 0.5, 0, 1]
        x = np.linspace(0, 100, 101)
        models.simple_harmonic_oscillator(ps, x)
        models.damped_harmonic_oscillator(pd, x)

    def test_acoustic_phonon(self):
        '''Test acoustic phonon dispersion
        '''
        p = [1, 0.5]
        x = np.linspace(0, 20, 101)
        models.acoustic_phonon_dispersion(p, x)

    def test_magnon_dispersion(self):
        '''Test ferro- and antiferro- magnon dispersions
        '''
        p = [1, 0.5]
        x = np.linspace(0, 1, 101)
        models.ferromagnetic_disperions(p, x)
        models.antiferromagnetic_disperions(p, x)


if __name__ == '__main__':
    unittest.main()
