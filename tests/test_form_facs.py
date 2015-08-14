from neutronpy import form_facs
import numpy as np
import unittest


class StructureFactor(unittest.TestCase):
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
        structure = form_facs.Material(self.input)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((2., 0., 0.))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((2, 0, 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((0, 2., 0))) ** 2, 1702170.4663405998, 6)
        self.assertAlmostEqual(np.abs(structure.calc_str_fac((0, 2, 0))) ** 2, 1702170.4663405998, 6)

        ndarray_example = np.linspace(0.5, 1.5, 21)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((ndarray_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, ndarray_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, ndarray_example))) ** 2), 16831011.814390473, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((ndarray_example, ndarray_example, 0))) ** 2), 10616602.544519115, 6)


        list_example = list(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((list_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, list_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, list_example))) ** 2), 16831011.814390473, 6)

        tuple_example = tuple(ndarray_example)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((tuple_example, 0, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, tuple_example, 0))) ** 2), 7058726.6759794801, 6)
        self.assertAlmostEqual(np.sum(abs(structure.calc_str_fac((0, 0, tuple_example))) ** 2), 16831011.814390473, 6)

    def test_N_atoms(self):
        structure = form_facs.Material(self.input)
        self.assertTrue(np.abs(structure.N_atoms(22.) - 3.611085294775001e+22) < 1)


class MagneticFormFactor(unittest.TestCase):
    def test_mag_form_fac(self):
        ion = form_facs.Ion('Fe')

        formfac, _temp = ion.calc_mag_form_fac(q=1.)[0], ion.calc_mag_form_fac(q=1.)[1:]

        self.assertAlmostEqual(formfac, 0.932565, 6)


if __name__ == "__main__":
    unittest.main()
