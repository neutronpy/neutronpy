from neutronpy import Energy, Data, load, save, detect_filetype, functions
import numpy as np
from scipy.integrate import simps
import os
import unittest


class EnergyTest(unittest.TestCase):
    def test_energy(self):
        energy = Energy(energy=25.)
        self.assertAlmostEqual(energy.energy, 25.0, 4)
        self.assertAlmostEqual(energy.wavelength, 1.8089, 4)
        self.assertAlmostEqual(energy.wavevector, 3.473, 3)
        self.assertAlmostEqual(energy.velocity, 2187., 0)
        self.assertAlmostEqual(energy.temperature, 290.113, 3)
        self.assertAlmostEqual(energy.frequency, 6.045, 3)


class DataTest(unittest.TestCase):
    def build_data(self, clean=True):
        p = np.array([20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2])
        x = np.linspace(-1, 1, 81)

        if clean:
            y = functions.voigt(p, x)
            mon = 1e5
            tim = 15
        else:
            y = functions.voigt(p, x) + np.random.normal(loc=0., scale=5, size=len(x))
            mon = 1e3
            tim = 5

        output = Data(Q=np.vstack((item.ravel() for item in np.meshgrid(x, 0., 0., 0., 300.))).T,
                      detector=y, monitor=np.full(x.shape, mon), time=np.full(x.shape, tim))

        return output

    def test_load_files(self):
        data1 = load((os.path.join(os.path.dirname(__file__), 'scan0001.dat'),
                      os.path.join(os.path.dirname(__file__), 'scan0002.dat')))
        data2 = load((os.path.join(os.path.dirname(__file__), 'scan0003.ng5')))
        data3 = load((os.path.join(os.path.dirname(__file__), 'scan0004.bt7')))

    def test_save_file(self):
        pass
    
    def test_filetype_detection(self):
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0001.dat')) == 'SPICE')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0003.ng5')) == 'ICP')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0004.bt7')) == 'ICE')
    
    def test_combine_data(self):
        data1 = self.build_data(clean=True)
        data2 = self.build_data(clean=False)

        data = data1 + data2

        self.assertTrue((data.monitor == data1.monitor + data2.monitor).all())
        self.assertTrue((data.detector == data1.detector + data2.detector).all())
        self.assertTrue((data.Q == data1.Q).all() & (data.Q == data2.Q).all())

    def test_rebin(self):
        data = self.build_data(clean=True)
        data_bin = data.bin(dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))

        self.assertEqual(data_bin.Q.shape[0], 41)
        self.assertEqual(data_bin.monitor.shape[0], 41)
        self.assertEqual(data_bin.detector.shape[0], 41)

        self.assertEqual(np.average(data_bin.monitor), np.average(data.monitor))
        self.assertTrue(abs(simps(data_bin.detector, data_bin.Q[:, 0]) - simps(data.detector, data.Q[:, 0])) <= 0.1)

    def test_analysis(self):
        data = self.build_data(clean=True)
        
        self.assertAlmostEqual(data.integrate(), 45.8424794006, 6)
        self.assertTrue((data.position()[0] < 1e-15))
        self.assertAlmostEqual(data.width()[0], 0.3, 2)


if __name__ == '__main__':
    unittest.main()
