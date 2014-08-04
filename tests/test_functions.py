from neutronpy import functions
import unittest
import numpy as np
from scipy.integrate import simps


class FunctionTests(unittest.TestCase):
    def test_gauss_norm(self):
        p = np.array([0., 0., 1., -30., 3., 1., 30., 3.])
        x = np.linspace(-1e6, 1e6, 8e6 + 1)
        y = functions.gaussian(p, x)
        integ = simps(y, x)
        self.assertAlmostEqual(integ, 2., 5)

    def test_lorent_norm(self):
        p = np.array([0., 0., 1., -30., 3., 1., 30., 3.])
        x = np.linspace(-1e6, 1e6, 8e6 + 1)
        y = functions.lorentzian(p, x)
        integ = simps(y, x)
        self.assertAlmostEqual(integ, 2., 5)

    def test_voigt_norm(self):
        p = np.array([0., 0., 1., -30., 2., 3., 1., 30., 2., 3.])
        x = np.linspace(-1e6, 1e6, 8e6 + 1)
        y = functions.voigt(p, x)
        integ = simps(y, x)
        self.assertAlmostEqual(integ, 2., 5)

    def test_gaussring_norm(self):
        p = np.array([0., 0., 1., 0., 0., 0.5, 0.5, 0.1])
        a, b = np.linspace(-10, 10, 1001), np.linspace(-10, 10, 1001)
        q = np.meshgrid(a, b, sparse=True)
        y = functions.gaussian_ring(p, q)
        integ = simps(simps(y, b), a)
        self.assertAlmostEqual(integ, 1., 5)

    def test_resolution_norm(self):
        p = np.array([0., 0., 1., 0., 0., 1.43, 23867.71, 22311.93, 20739.82])
        a, b = np.linspace(-1, 1, 501), np.linspace(-1, 1, 501)
        q = np.meshgrid(a, b, sparse=True)
        y = functions.resolution(p, q)
        integ = simps(simps(y, b), a)
        self.assertAlmostEqual(integ, 1., 5)


if __name__ == '__main__':
    unittest.main()
