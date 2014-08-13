from neutronpy import tools, functions
import numpy as np
from scipy.integrate import simps
import os
import unittest


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
    def build_data(self, clean=True):
        p = np.array([20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2])
        x = np.linspace(-1, 1, 81)

        if clean:
            y = functions.voigt(p, x)
            mon = 1e5
        else:
            y = functions.voigt(p, x) + np.random.normal(loc=0., scale=5, size=len(x))
            mon = 1e3

        output = tools.Data()
        output.Q = np.vstack((item.ravel() for item in np.meshgrid(x, 0., 0., 0.))).T
        output.detector = y
        output.monitor = np.ones(x.shape) * mon
        output.temp = np.ones(x.shape) * 300.

        return output

    def test_load_files(self):
        data1 = tools.Data()
        data1.load_file(os.path.join(os.path.dirname(__file__), 'scan0001.dat'), os.path.join(os.path.dirname(__file__), 'scan0002.dat'), mode='HFIR')

        data2 = tools.Data()
        data2.load_file(os.path.join(os.path.dirname(__file__), 'scan0003.ng5'), mode='NCNR')

    def test_combine_data(self):
        data1 = self.build_data(clean=True)
        data2 = self.build_data(clean=False)

        data = data1 + data2

        self.assertTrue((data.monitor == data1.monitor + data2.monitor).all())
        self.assertTrue((data.detector == data1.detector + data2.detector).all())
        self.assertTrue((data.temp == data1.temp).all() & (data.temp == data2.temp).all())
        self.assertTrue((data.Q == data1.Q).all() & (data.Q == data2.Q).all())

    def test_rebin(self):
        data = self.build_data(clean=True)
        Q, monitor, detector, temps = data.bin([-1, 1., 41], [-0.1, 0.1, 1], [-0.1, 0.1, 1], [3.5, 4.5, 1], [-300, 900, 1])

        self.assertEqual(Q.shape[0], 41)
        self.assertEqual(monitor.shape[0], 41)
        self.assertEqual(detector.shape[0], 41)
        self.assertEqual(temps.shape[0], 41)

        self.assertEqual(np.average(monitor), np.average(data.monitor))
        self.assertTrue(abs(simps(detector, Q[:, 0]) - simps(data.detector, data.Q[:, 0])) <= 0.1)

    def test_analysis(self):
        data = self.build_data(clean=True)

        self.assertAlmostEqual(data.integrate(), 45.8424794006, 6)
        self.assertTrue((data.position()[0] < 1e-15))
        self.assertAlmostEqual(data.width()[0], 0.3, 2)


if __name__ == '__main__':
    unittest.main()
