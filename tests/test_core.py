from neutronpy import Energy, Data, load, save, detect_filetype, functions
from neutronpy.constants import BOLTZMANN_IN_MEV_K
import numpy as np
from scipy.integrate import simps
import os
import unittest
from unittest.mock import patch
from copy import deepcopy


class EnergyTest(unittest.TestCase):
    def test_energy(self):
        energy = Energy(energy=25.)
        self.assertEqual(np.round(energy.energy), 25.0)
        self.assertEqual(np.round(energy.wavelength, 4), 1.8089)
        self.assertEqual(np.round(energy.wavevector, 3), 3.473)
        self.assertEqual(np.round(energy.velocity, 3), 2186.967)
        self.assertEqual(np.round(energy.temperature, 3), 290.113)
        self.assertEqual(np.round(energy.frequency, 3), 6.045)

        stringtest = u'\nEnergy: {0:3.3f} meV'
        stringtest += u'\nWavelength: {1:3.3f} Å'
        stringtest += u'\nWavevector: {2:3.3f} 1/Å'
        stringtest += u'\nVelocity: {3:3.3f} m/s'
        stringtest += u'\nTemperature: {4:3.3f} K'
        stringtest += u'\nFrequency: {5:3.3f} THz\n'
        stringtest = stringtest.format(25.0, 1.8089, 3.473, 2186.967,
                                       290.113, 6.045)

        self.assertTrue(energy.values == stringtest)

    def test_energy_setters(self):
        energy = Energy(energy=25.)
        energy.energy = 25
        self.assertEqual(np.round(energy.wavelength, 4), 1.8089)
        energy.wavelength = 1.8
        self.assertEqual(np.round(energy.energy, 1), 25.2)
        energy.wavevector = 3.5
        self.assertEqual(np.round(energy.energy, 1), 25.4)
        energy.velocity = 2180
        self.assertEqual(np.round(energy.energy, 1), 24.8)
        energy.temperature = 290
        self.assertEqual(np.round(energy.energy, 1), 25.0)
        energy.frequency = 6
        self.assertEqual(np.round(energy.energy, 1), 24.8)


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
                      detector=y, monitor=np.full(x.shape, mon, dtype=float), time=np.full(x.shape, tim, dtype=float))

        return output

    def build_3d_data(self):
        p = np.array([0, 0, 1, 0, 0, 0.1])
        x, y = np.linspace(-1, 1, 81), np.linspace(-1, 1, 81)
        X, Y = np.meshgrid(x, y)
        z = functions.gaussian2d(p, (X, Y))
        mon = 1e5
        tim = 15

        output = Data(Q=np.vstack((item.ravel() for item in np.meshgrid(x, y, 0., 0., 300.))).T,
                      detector=z.ravel(), monitor=np.full(X.ravel().shape, mon, dtype=float),
                      time=np.full(X.ravel().shape, tim, dtype=float))

        return output

    @patch('sys.stdout')
    def test_load_files(self, mock_stdout):
        data1 = load((os.path.join(os.path.dirname(__file__), 'scan0001.dat'),
                      os.path.join(os.path.dirname(__file__), 'scan0002.dat')))
        data2 = load((os.path.join(os.path.dirname(__file__), 'scan0003.ng5')))
        data3 = load((os.path.join(os.path.dirname(__file__), 'scan0004.bt7')))
        data4 = load((os.path.join(os.path.dirname(__file__), 'scan0007.bt7')))
        data5 = load((os.path.join(os.path.dirname(__file__), 'scan0005')))
        with self.assertRaises(ValueError):
            load((os.path.join(os.path.dirname(__file__), 'scan0006.test')), filetype='blah')

    def test_save_file(self):
        data_out = self.build_data()
        save(data_out, 'test.out', fileformat='ascii')
        save(data_out, 'test.out', fileformat='hdf5')
        save(data_out, 'test.out', fileformat='pickle')
        with self.assertRaises(ValueError):
            save(data_out, 'test.out', fileformat='blah')

    def test_filetype_detection(self):
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0001.dat')) == 'SPICE')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0003.ng5')) == 'ICP')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0004.bt7')) == 'ICE')
        self.assertTrue(detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0005')) == 'MAD')
        with self.assertRaises(ValueError):
            detect_filetype(os.path.join(os.path.dirname(__file__), 'scan0006.test'))

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

        bounds = (data.h >= -1) & (data.h <= 1)
        self.assertAlmostEqual(data.integrate(bounds=bounds)[0], 45.8424794006, 6)
        self.assertTrue((data.position(bounds=bounds)[0] < 1e-15))
        self.assertEqual(np.round(data.width(bounds=bounds)[0], 1), 0.3)

        background = dict(type='constant', value=0.)
        self.assertAlmostEqual(data.integrate(background=background), 45.8424794006, 6)
        self.assertTrue((data.position(background=background)[0] < 1e-15))
        self.assertEqual(np.round(data.width(background=background)[0], 1), 0.3)

        background = dict(type='percent', value=2)
        self.assertAlmostEqual(data.integrate(background=background), 5.6750023056707004, 6)
        background = dict(type='minimum')
        self.assertAlmostEqual(data.integrate(background=background), 5.6750023056707004, 6)
        background = dict(type='blah')
        self.assertAlmostEqual(data.integrate(background=background), 45.8424794006, 6)

    def test_init_cases(self):
        data = Data()

    def test_add_sub_data(self):
        data1 = Data()
        del data1.Q
        data2 = self.build_data()
        with self.assertRaises(AttributeError):
            data1 + data2
        with self.assertRaises(AttributeError):
            data1 - data2

    def test_mul_div_pow_data(self):
        data = self.build_data()
        data1 = deepcopy(data) * 10.
        data2 = deepcopy(data) / 10.
        data3 = deepcopy(data) // 10.
        data4 = deepcopy(data) ** 10.

        self.assertTrue(np.all(data1.detector == data.detector * 10))
        self.assertTrue(np.all(data2.detector == data.detector / 10))
        self.assertTrue(np.all(data3.detector == data.detector // 10))
        self.assertTrue(np.all(data4.detector == data.detector ** 10))

    def test_setters(self):
        data = self.build_data()
        data.h = 3
        with self.assertRaises(ValueError):
            data.h = np.zeros(5)
        data.k = 3
        with self.assertRaises(ValueError):
            data.k = np.zeros(5)
        data.l = 3
        with self.assertRaises(ValueError):
            data.l = np.zeros(5)
        data.e = 3
        with self.assertRaises(ValueError):
            data.e = np.zeros(5)
        data.temp = 3
        with self.assertRaises(ValueError):
            data.temp = np.zeros(5)

    def test_norms(self):
        data = self.build_data()
        data.m0 = 0
        self.assertTrue(np.all(data.intensity == data.detector / data.monitor * data.m0))
        data.time_norm = True
        data.t0 = 0
        self.assertTrue(np.all(data.intensity == data.detector / data.time * data.t0))

    def test_error(self):
        data = self.build_data()
        data._err = np.sqrt(data.detector)
        data.m0 = 0
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))
        data._err = None
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))
        del data._err
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))

        data.time_norm = True
        data._err = np.sqrt(data.detector)
        data.t0 = 0
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))
        data._err = None
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))
        del data._err
        self.assertTrue(np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))

        data.error = 10
        with self.assertRaises(ValueError):
            data.error = np.zeros(5)

    def test_detailed_balance(self):
        data = self.build_data()
        self.assertTrue(np.all(data.detailed_balance_factor == 1. - np.exp(-data.e / BOLTZMANN_IN_MEV_K / data.temp)))

    def test_scattering_function(self):
        from neutronpy.form_facs import Material
        input = {'name': 'FeTe',
                 'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                 {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                 {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                 {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                 'debye-waller': True,
                 'massNorm': True,
                 'formulaUnits': 1.,
                 'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

        data = self.build_data()
        material = Material(input)

        ki = Energy(energy=14.7).wavevector
        kf = Energy(energy=14.7 - data.e).wavevector

        self.assertTrue(np.all(data.scattering_function(material, 14.7) == 4 * np.pi / (material.total_scattering_cross_section) * ki / kf * data.detector))

    def test_dynamic_susceptibility(self):
        from neutronpy.form_facs import Material
        input = {'name': 'FeTe',
                 'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                 {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                 {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                 {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                 'debye-waller': True,
                 'massNorm': True,
                 'formulaUnits': 1.,
                 'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

        data = self.build_data()
        material = Material(input)

        ki = Energy(energy=14.7).wavevector
        kf = Energy(energy=14.7 - data.e).wavevector

        self.assertTrue(np.all(data.dynamic_susceptibility(material, 14.7) == 4 * np.pi / (material.total_scattering_cross_section) * ki / kf *
                               data.detector * data.detailed_balance_factor))

    def test_background_subtraction(self):
        data = self.build_data(clean=True)
        background_data = self.build_data(clean=False)
        data.subtract_background(background_data, ret=False)

    @patch("matplotlib.pyplot.show")
    def test_plotting(self, mock_show):
        data = self.build_data()
        data.plot('h', 'intensity')
        data.plot('h', 'intensity', show_err=False)
        data.plot('h', 'intensity', output_file='plot_test.pdf')
        data.plot('h', 'intensity', smooth_options=dict(sigma=1))
        data.plot('h', 'intensity', fit_options=dict(function=functions.voigt, fixp=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], p=[20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2]))
        data.plot('h', 'intensity', to_bin=dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))
        data3d = self.build_3d_data()
        data3d.plot(x='h', y='k', z='intensity')
        with self.assertRaises(KeyError):
            data3d.plot('h', 'k', 'blah')
        with self.assertRaises(KeyError):
            data3d.plot('h', 'k', 'w', 'blah')

if __name__ == '__main__':
    unittest.main()
