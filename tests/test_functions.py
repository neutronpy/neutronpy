from neutronpy import functions
import unittest
import numpy as np


class FunctionTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(FunctionTests, self).__init__(*args, **kwargs)
        self.x = np.linspace(-100000, 100000, 4000001)

    def test_gauss_norm(self):
        p = np.array([0., 0., 1., -30., 3., 1., 30., 3.])
        y = functions.gaussian(p, self.x)
        integ = np.trapz(y, x=self.x)
        self.assertAlmostEqual(integ, 2., 4)

    def test_lorent_norm(self):
        p = np.array([0., 0., 1., -30., 3., 1., 30., 3.])
        y = functions.lorentzian(p, self.x)
        integ = np.trapz(y, x=self.x)
        self.assertAlmostEqual(integ, 2., 4)

    def test_voigt_norm(self):
        p = np.array([0., 0., 1., -30., 2., 3., 1., 30., 2., 3.])
        y = functions.voigt(p, self.x)
        integ = np.trapz(y, x=self.x)
        self.assertAlmostEqual(integ, 2., 4)

    def test_gaussring_norm(self):
        pass

    def test_resolution_norm(self):
        p = np.array([0., 0., 1., 0., 0., 1.43, 23867.71, 22311.93, 20739.82])
        q = np.meshgrid(np.linspace(-10, 10, 2001), np.linspace(-10, 10, 2001))
        y = functions.resolution(p, q)
        integ = np.trapz(y.flatten(), x=q[0].flatten()) + np.trapz(y.flatten(), x=q[1].flatten())
        self.assertAlmostEqual(integ, 1., 4)


if __name__ == '__main__':
    unittest.main()
