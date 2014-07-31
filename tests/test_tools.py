import unittest
from neutronpy import tools


class EnergyTest(unittest.TestCase):

    def test_energy(self):
        energy = tools.Neutron(e=25.)
        self.assertAlmostEqual(energy.e, 25.0, 4)
        self.assertAlmostEqual(energy.l, 1.8089, 4)
        self.assertAlmostEqual(energy.k, 3.473, 3)
        self.assertAlmostEqual(energy.v, 2187., 0)
        self.assertAlmostEqual(energy.temp, 290.113, 3)
        self.assertAlmostEqual(energy.freq, 6.045, 3)


class DataTest(unittest.TestCase):

    def test_load_file(self):
        pass

    def test_combine_data(self):
        pass

    def test_rebin(self):
        pass

    def test_analysis(self):
        pass


if __name__ == '__main__':
    unittest.main()
