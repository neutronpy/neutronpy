# -*- coding: utf-8 -*-
r'''Tests lattice math

'''
import unittest
import numpy as np
from neutronpy.crystal import lattice


class LatticeTests(unittest.TestCase):
    '''Unit Tests for Lattice operations
    '''
    def __init__(self, *args, **kwargs):
        '''Initialize unit tests
        '''
        super(LatticeTests, self).__init__(*args, **kwargs)

        self.unitcell = lattice.Lattice(4, 4, 4, 90, 90, 90)

    def test_get_angle_between_planes(self):
        '''Tests get angle between planes defined by two vectors
        '''
        self.assertTrue(self.unitcell.get_angle_between_planes([1, 0, 0], [1, 1, 1]) - 54.73561031724535 < 1e-6)

    def test_get_d_spacing(self):
        '''Tests d-spacing for given HKL
        '''
        self.assertTrue(self.unitcell.get_d_spacing([1, 1, 1]) == self.unitcell.a / np.sqrt(3))

    def test_get_q(self):
        '''Tests q for given HKL
        '''
        self.assertAlmostEqual(self.unitcell.get_q([1, 1, 1]), 2 * np.pi / self.unitcell.b * np.sqrt(3), 12)

    def test_get_two_theta(self):
        '''Tests 2theta for given HKL
        '''
        self.assertTrue(self.unitcell.get_two_theta([1, 1, 1], 2) == 51.317812546510552)

    def test_constants(self):
        '''Test that gettters/setters work properly
        '''
        abg = np.array([self.unitcell.alpha, self.unitcell.beta, self.unitcell.gamma])
        abc = np.array([self.unitcell.a, self.unitcell.b, self.unitcell.c])
        abg_rad = np.array([self.unitcell.alpha_rad, self.unitcell.beta_rad, self.unitcell.gamma_rad])
        abgstar_rad = np.array([self.unitcell.alphastar_rad, self.unitcell.betastar_rad, self.unitcell.gammastar_rad])
        abgstar = np.array([self.unitcell.alphastar, self.unitcell.betastar, self.unitcell.gammastar])
        abcstar = np.array([self.unitcell.astar, self.unitcell.bstar, self.unitcell.cstar])

        self.assertTrue(np.all(self.unitcell.abc == abc))
        self.assertTrue(np.all(self.unitcell.abg == abg))
        self.assertTrue(np.all(self.unitcell.abg_rad == abg_rad))
        self.assertTrue(np.all(self.unitcell.abg_rad == np.deg2rad(abg)))
        self.assertEqual(np.round(self.unitcell.volume, 12), 4 ** 3)
        self.assertEqual(np.round(self.unitcell.reciprocal_volume, 12), 1. / (4 ** 3))
        self.assertTrue(self.unitcell.astar == self.unitcell.b * self.unitcell.c * np.sin(self.unitcell.alpha_rad) / self.unitcell.volume)
        self.assertTrue(self.unitcell.bstar == self.unitcell.a * self.unitcell.c * np.sin(self.unitcell.beta_rad) / self.unitcell.volume)
        self.assertTrue(self.unitcell.cstar == self.unitcell.a * self.unitcell.b * np.sin(self.unitcell.gamma) / self.unitcell.volume)
        self.assertTrue(np.all(abgstar_rad == np.deg2rad(abgstar)))
        self.assertTrue(np.all(self.unitcell.reciprocal_abc == abcstar))
        self.assertTrue(np.all(self.unitcell.reciprocal_abg == abgstar))
        self.assertTrue(np.all(self.unitcell.reciprocal_abg_rad == abgstar_rad))
        self.assertTrue(np.all(np.round(self.unitcell.Bmatrix * self.unitcell.Bmatrix.T, 12) == np.round(self.unitcell.Gstar, 12)))

    def test_lattice_type(self):
        '''Test lattice type determination
        '''
        test_cell = self.unitcell
        self.assertTrue(test_cell.lattice_type == 'cubic')

        test_cell = lattice.Lattice(1, 1, 2, 90, 90, 90)
        self.assertTrue(test_cell.lattice_type == 'tetragonal')

        test_cell = lattice.Lattice(1, 2, 3, 90, 90, 90)
        self.assertTrue(test_cell.lattice_type == 'orthorhombic')

        test_cell = lattice.Lattice(1, 2, 3, 90, 89, 90)
        self.assertTrue(test_cell.lattice_type == 'monoclinic')

        test_cell = lattice.Lattice(1, 1, 1, 39, 39, 39)
        self.assertTrue(test_cell.lattice_type == 'rhombohedral')

        test_cell = lattice.Lattice(1, 1, 1, 39, 39, 39)
        self.assertTrue(test_cell.lattice_type == 'rhombohedral')

        test_cell = lattice.Lattice(1, 1, 2, 90, 90, 120)
        self.assertTrue(test_cell.lattice_type == 'hexagonal')

        test_cell = lattice.Lattice(1, 2, 3, 30, 60, 120)
        self.assertTrue(test_cell.lattice_type == 'triclinic')

        test_cell = lattice.Lattice(1, 1, 2, 90, 90, 150)
        self.assertRaises(ValueError, getattr, test_cell, 'lattice_type')


class GoniometerTests(unittest.TestCase):
    '''Unit Tests for goniometer
    '''
    def test_goniometer_constants(self):
        '''Test constants
        '''
        pass


if __name__ == "__main__":
    unittest.main()
