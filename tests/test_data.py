# -*- coding: utf-8 -*-
r'''Testing of core library

'''
from copy import deepcopy
import os
import unittest
from mock import patch
from matplotlib import use
import numpy as np
from scipy.integrate import simps
from neutronpy import Energy, Data, functions
from neutronpy.io import load_data, save_data, detect_filetype
from neutronpy.constants import BOLTZMANN_IN_MEV_K
use('Agg')


def build_data(clean=True):
    '''Builds data object
    '''
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


def build_3d_data():
    '''Builds 3D data object
    '''
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


class DataTest(unittest.TestCase):
    '''Unit tests for data object
    '''
    def test_combine_data(self):
        '''Tests combining data
        '''
        data1 = build_data(clean=True)
        data2 = build_data(clean=False)

        data = data1 + data2

        self.assertTrue((data.monitor == data1.monitor + data2.monitor).all())
        self.assertTrue((data.detector == data1.detector + data2.detector).all())
        self.assertTrue((data.Q == data1.Q).all() & (data.Q == data2.Q).all())

    def test_rebin(self):
        '''Tests data rebinning
        '''
        data = build_data(clean=True)
        data_bin = data.bin(dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))

        self.assertEqual(data_bin.Q.shape[0], 41)
        self.assertEqual(data_bin.monitor.shape[0], 41)
        self.assertEqual(data_bin.detector.shape[0], 41)

        self.assertEqual(np.average(data_bin.monitor), np.average(data.monitor))
        self.assertTrue(abs(simps(data_bin.detector, data_bin.Q[:, 0]) - simps(data.detector, data.Q[:, 0])) <= 0.1)

    def test_analysis(self):
        '''Tests analysis methods
        '''
        data = build_data(clean=True)

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
        '''Tests initialization cases
        '''
        try:
            Data()
        except:
            self.fail('Data initialization failed')

    def test_add_sub_data(self):
        '''Tests adding and subtracting data
        '''
        data1 = Data()
        del data1.Q
        data2 = build_data()

        def _test(test):
            if test == 'add':
                data1 + data2
            elif test == 'sub':
                data1 - data2

        self.assertRaises(AttributeError, _test, 'add')
        self.assertRaises(AttributeError, _test, 'sub')

    def test_mul_div_pow_data(self):
        '''Tests multiplication, division, and power operations on data object
        '''
        data = build_data()
        data1 = deepcopy(data) * 10.
        data2 = deepcopy(data) / 10.
        data3 = deepcopy(data) // 10.
        data4 = deepcopy(data) ** 10.

        self.assertTrue(np.all(data1.detector == data.detector * 10))
        self.assertTrue(np.all(data2.detector == data.detector / 10))
        self.assertTrue(np.all(data3.detector == data.detector // 10))
        self.assertTrue(np.all(data4.detector == data.detector ** 10))

    def test_setters(self):
        '''Tests setters working
        '''
        def _test(test):
            data = build_data()
            if test == 'h':
                data.h = 3
                data.h = np.zeros(5)
            elif test == 'k':
                data.k = 3
                data.k = np.zeros(5)
            elif test == 'l':
                data.l = 3
                data.l = np.zeros(5)
            elif test == 'e':
                data.e = 3
                data.e = np.zeros(5)
            elif test == 'temp':
                data.temp = 3
                data.temp = np.zeros(5)
        self.assertRaises(ValueError, _test, 'h')
        self.assertRaises(ValueError, _test, 'k')
        self.assertRaises(ValueError, _test, 'l')
        self.assertRaises(ValueError, _test, 'e')
        self.assertRaises(ValueError, _test, 'temp')

    def test_norms(self):
        '''Test normalization types
        '''
        data = build_data()
        data.m0 = 0
        self.assertTrue(np.all(data.intensity == data.detector / data.monitor * data.m0))
        data.time_norm = True
        data.t0 = 0
        self.assertTrue(np.all(data.intensity == data.detector / data.time * data.t0))

    def test_error(self):
        '''Tests exception handling
        '''
        data = build_data()
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

        def _test(test):
            data.error = 10
            data.error = np.zeros(5)

        self.assertRaises(ValueError, _test, None)

    def test_detailed_balance(self):
        '''Test detailed balance factor
        '''
        data = build_data()
        self.assertTrue(np.all(data.detailed_balance_factor == 1. - np.exp(-data.e / BOLTZMANN_IN_MEV_K / data.temp)))

    def test_scattering_function(self):
        '''Test scattering function
        '''
        from neutronpy.material import Material
        input_mat = {'name': 'FeTe',
                     'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                     {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                     {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                     {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                     'debye-waller': True,
                     'massNorm': True,
                     'formulaUnits': 1.,
                     'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

        data = build_data()
        material = Material(input_mat)

        ki = Energy(energy=14.7).wavevector
        kf = Energy(energy=14.7 - data.e).wavevector

        self.assertTrue(np.all(data.scattering_function(material, 14.7) == 4 * np.pi / (material.total_scattering_cross_section) * ki / kf * data.detector))

    def test_dynamic_susceptibility(self):
        '''Test dynamic susceptibility
        '''
        from neutronpy.material import Material
        input_mat = {'name': 'FeTe',
                     'composition': [{'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                                     {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                                     {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                                     {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]}],
                     'debye-waller': True,
                     'massNorm': True,
                     'formulaUnits': 1.,
                     'lattice': dict(abc=[3.81, 3.81, 6.25], abg=[90, 90, 90])}

        data = build_data()
        material = Material(input_mat)

        ki = Energy(energy=14.7).wavevector
        kf = Energy(energy=14.7 - data.e).wavevector

        self.assertTrue(np.all(data.dynamic_susceptibility(material, 14.7) == 4 * np.pi / (material.total_scattering_cross_section) * ki / kf *
                               data.detector * data.detailed_balance_factor))

    def test_background_subtraction(self):
        '''Test background subtraction
        '''
        data = build_data(clean=True)
        background_data = build_data(clean=False)
        try:
            data.subtract_background(background_data, ret=False)
        except:
            self.fail('background subtraction failed')

    @patch("matplotlib.pyplot.show")
    def test_plotting(self, mock_show):
        '''Test plotting
        '''
        data = build_data()
        data.plot('h', 'intensity')
        data.plot('h', 'intensity', show_err=False)
        data.plot('h', 'intensity', output_file='plot_test.pdf')
        data.plot('h', 'intensity', smooth_options=dict(sigma=1))
        data.plot('h', 'intensity', fit_options=dict(function=functions.voigt, fixp=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], p=[20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2]))
        data.plot('h', 'intensity', to_bin=dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))
        data3d = build_3d_data()
        data3d.plot(x='h', y='k', z='intensity')
        self.assertRaises(KeyError, data3d.plot, 'h', 'k', 'blah')
        self.assertRaises(KeyError, data3d.plot, 'h', 'k', 'w', 'blah')


if __name__ == '__main__':
    unittest.main()
