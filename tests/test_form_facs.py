from neutronpy import form_facs
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
                      'lattice': [3.81, 3.81, 6.25]}

    def test_str_fac(self):
        structure = form_facs.Material(self.input)
        self.assertAlmostEqual(abs(structure.calc_str_fac((2., 0., 0.)) ** 2), 1583878.155915682, 6)


class MagneticFormFactor(unittest.TestCase):
    def test_mag_form_fac(self):
        ion = form_facs.Ion('Fe')

        formfac, _temp = ion.calc_mag_form_fac(q=1.)[0], ion.calc_mag_form_fac(q=1.)[1:]

        self.assertAlmostEqual(formfac, 0.932565, 6)


if __name__ == "__main__":
    unittest.main()
