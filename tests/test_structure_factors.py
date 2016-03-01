# -*- coding: utf-8 -*-
r'''Test structure factor calculations

'''
import unittest
import numpy as np
from mock import patch
from matplotlib import use
from neutronpy.material import Material
from neutronpy.structure_factors import MagneticFormFactor
use('Agg')


class StructureFactor(unittest.TestCase):
    '''Unit tests for Structure Factor
    '''
    def __init__(self, *args, **kwargs):
        super(StructureFactor, self).__init__(*args, **kwargs)

        self.input = {'name': 'FeTe',
                      'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                      {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                      {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                      {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                      'debye-waller': True,
                      'massNorm': True,
                      'formulaUnits': 1.,
                      'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

    def test_str_fac(self):
        '''Tests various positions for structure factor
        '''
        structure = Material(self.input)
        self.assertAlmostEqual(np.abs(structure.calc_nuc_str_fac((2., 0., 0.))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_nuc_str_fac((2, 0, 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_nuc_str_fac((0, 2., 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_nuc_str_fac((0, 2, 0))) ** 2, 1702170.4663405998, 6)

        ndarray_example = np.linspace(0.5, 1.5, 21)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((ndarray_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, ndarray_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, 0, ndarray_example))) ** 2), 16831011.814390473, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((ndarray_example, ndarray_example, 0))) ** 2), 10616602.544519115, 6)

        list_example = list(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((list_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, list_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, 0, list_example))) ** 2), 16831011.814390473, 6)

        tuple_example = tuple(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((tuple_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, tuple_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_nuc_str_fac((0, 0, tuple_example))) ** 2), 16831011.814390473, 6)

    def test_N_atoms(self):
        '''Tests number of atoms in X g of material
        '''
        structure = Material(self.input)
        self.assertTrue(structure.N_atoms(22) == 36110850351331465494528)

    def test_volume(self):
        '''Tests volume of unitcell
        '''
        structure = Material(self.input)
        self.assertTrue(structure.volume == 90.725624999999965)

    def test_total_scattering_cross_section(self):
        '''Tests scattering cross section
        '''
        structure = Material(self.input)
        self.assertTrue(structure.total_scattering_cross_section == 31.880000000000003)

    def test_case(self):
        '''Test formulaUnits
        '''
        input_test = self.input
        del input_test['formulaUnits']
        structure = Material(input_test)
        del structure

    @patch("matplotlib.pyplot.show")
    def test_plot(self, mock_show):
        '''Test unitcell plot
        '''
        structure = Material(self.input)
        structure.plot_unit_cell()

    def test_optimal_thickness(self):
        '''Test optimal thickness calculation
        '''
        structure = Material(self.input)
        self.assertEqual(structure.calc_optimal_thickness(), 1.9552936422413782)


class MagneticFormFactorTest(unittest.TestCase):
    '''Unit tests for magnetic form factor
    '''
    def test_mag_form_fac(self):
        '''Tests the magnetic form factor single value
        '''
        ion = MagneticFormFactor('Fe')
        formfac, _temp = ion.calc_mag_form_fac(q=1.)[0], ion.calc_mag_form_fac(q=1.)[1:]
        del _temp
        self.assertAlmostEqual(formfac, 0.932565, 6)

    def test_mag_form_fac_case1(self):
        '''Tests the magnetic form factor no q given
        '''
        ion = MagneticFormFactor('Fe')
        formfac, _temp = ion.calc_mag_form_fac()[0], ion.calc_mag_form_fac()[1:]
        del _temp
        self.assertAlmostEqual(np.sum(formfac), 74.155233575216599, 12)

    def test_mag_form_fac_case2(self):
        '''Tests the magnetic form factor q range provided
        '''
        ion = MagneticFormFactor('Fe')
        formfac, _temp = ion.calc_mag_form_fac(qrange=[0, 2])[0], ion.calc_mag_form_fac(qrange=[0, 2])[1:]
        del _temp
        self.assertAlmostEqual(np.sum(formfac), 74.155233575216599, 12)


if __name__ == "__main__":
    unittest.main()
