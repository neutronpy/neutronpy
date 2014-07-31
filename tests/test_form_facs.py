from neutronpy import form_facs
import unittest


class StructureFactor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(StructureFactor, self).__init__(*args, **kwargs)

        self.input = {'name': 'Fe1.1Te',
                      'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                      {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                      {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                      {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]},
                                      {'ion': 'Fe', 'pos': [0.25, 0.25, 0.721], 'occupancy': 0.1},
                                      {'ion': 'Fe', 'pos': [1. - 0.25, 1. - 0.25, 1. - 0.721], 'occupancy': 0.1}],
                      'debye-waller': False,
                      'massNorm': True,
                      'formulaUnits': 2.,
                      'lattice': [3.81, 3.81, 6.25]}

    def test_str_fac(self):
        structure = form_facs.Material(self.input)
        self.assertAlmostEqual(abs(structure.calc_str_fac((1., 1., 0.)) ** 2), 892301.46253218898, 6)


class MagneticFormFactor(unittest.TestCase):
    def test_mag_form_fac(self):
        ion = form_facs.Ion('Fe')

        formfac, *_temp = ion.calc_mag_form_fac(q=1.)
        self.assertAlmostEqual(formfac, 0.932565447328, 6)


if __name__ == "__main__":
    unittest.main()
