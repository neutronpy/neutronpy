# -*- coding: utf-8 -*-
r"""Testing of core library

"""
from copy import deepcopy

import numpy as np
import pytest
from matplotlib import use
from mock import patch
from neutronpy import Data, Energy, functions
from neutronpy.constants import BOLTZMANN_IN_MEV_K
from scipy.integrate import simps

use('Agg')


def build_data(clean=True):
    """Builds data object
    """
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

    output = Data(Q=np.vstack((item.ravel() for item in np.meshgrid(x, 0., 0., 4., 300.))).T,
                  detector=y, monitor=np.full(x.shape, mon, dtype=float), time=np.full(x.shape, tim, dtype=float))

    return output


def build_3d_data():
    """Builds 3D data object
    """
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


def test_build_data():
    data = Data(blah='test')
    data = Data(error=1)


def test_combine_data():
    """Tests combining data
    """
    data1 = build_data(clean=True)
    data2 = build_data(clean=False)
    data3 = object

    data = data1 + data2

    assert ((data.monitor == data1.monitor + data2.monitor).all())
    assert ((data.detector == data1.detector + data2.detector).all())
    assert ((data.Q == data1.Q).all() & (data.Q == data2.Q).all())

    def _test():
        data1 + data3

    with pytest.raises(TypeError):
        _test()


def test_rebin():
    """Tests data rebinning
    """
    data = Data(h=np.linspace(0, 1, 101), k=0, l=0, e=0, temp=0,
                detector=functions.gaussian([0, 0, 10, 0.5, 0.5], np.linspace(0, 1, 101)),
                monitor=np.ones(101), time=np.ones(101))

    data_bin = data.bin(dict(h=[0, 1., 51], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[-0.5, 0.5, 1]))

    assert (data_bin.Q.shape[0] == 51)
    assert (data_bin.monitor.shape[0] == 51)
    assert (data_bin.detector.shape[0] == 51)

    assert (np.average(data_bin.monitor) == np.average(data.monitor))
    assert (abs(simps(data_bin.detector, data_bin.Q[:, 0]) - simps(data.detector, data.Q[:, 0])) <= 0.1)
    assert (np.abs(data_bin.integrate() - data.integrate()) < 1e-1)
    assert (np.abs(data_bin.position()[0] - data.position()[0]) < 1e-1)
    assert (np.abs(data_bin.width()[0] - data.width()[0]) < 1e-1)

    def _test():
        data_bin = data.bin(dict(blah=[1, 2, 4]))

    with pytest.raises(KeyError):
        _test()


def test_analysis():
    """Tests analysis methods
    """
    x = np.linspace(-2, 2, 100)
    y = functions.gaussian([0, 0, 1, 0, 0.5], x)

    data = Data(Q=np.vstack((item.ravel() for item in np.meshgrid(x, 0., 0., 4., 300.))).T,
                detector=y, monitor=np.full(x.shape, 1, dtype=float), time=np.full(x.shape, 1, dtype=float))

    assert (np.abs(data.integrate() - 1) < 1e-5)
    assert (np.abs(data.position()[0]) < 1e-5)
    assert (abs(data.width(fwhm=True)[0] - 0.5) < 1e-1)

    assert isinstance(data.position(hkle=False), dict)
    assert isinstance(data.width(hkle=False), dict)

    assert (np.abs(data.integrate(hkle=False) - 1) < 1e-5)
    assert (np.abs(data.position(hkle=False)['h']) < 1e-5)
    assert (abs(data.width(fwhm=True, hkle=False)['h'] - 0.5) < 1e-1)

    bounds = (data.h >= -1) & (data.h <= 1)
    assert (np.abs(data.integrate(bounds=bounds) - 1) < 1e-5)
    assert (np.abs(data.position(bounds=bounds)[0]) < 1e-5)
    assert (abs(data.width(bounds=bounds, fwhm=True)[0] - 0.5) < 1e-1)

    background = dict(type='constant', value=0.)
    assert (np.abs(data.integrate(background=background) - 1) < 1e-5)
    assert (np.abs(data.position(background=background)[0]) < 1e-5)
    assert (abs(data.width(background=background, fwhm=True)[0] - 0.5) < 1e-1)

    background = dict(type='percent', value=2)
    assert (np.abs(data.integrate(background=background) - 1) < 1e-5)
    background = dict(type='minimum')
    assert (np.abs(data.integrate(background=background) - 1) < 1e-5)
    background = dict(type='blah')
    assert (np.abs(data.integrate(background=background) - 1) < 1e-5)


def test_init_cases():
    """Tests initialization cases
    """
    try:
        Data()
    except:
        pytest.fail('Data initialization failed')


def test_add_sub_data():
    """Tests adding and subtracting data
    """
    data1 = build_data()
    data2 = object

    def _test(test):
        if test == 'add':
            data1 + data2
        elif test == 'sub':
            data1 - data2

    with pytest.raises(TypeError):
        _test('add')
        _test('sub')


def test_mul_div_pow_data():
    """Tests multiplication, division, and power operations on data object
    """
    data = build_data()
    data1 = deepcopy(data) * 10.
    data2 = deepcopy(data) / 10.
    data3 = deepcopy(data) // 10.
    data4 = deepcopy(data) ** 10.

    assert (np.all(data1.detector == data.detector * 10))
    assert (np.all(data2.detector == data.detector / 10))
    assert (np.all(data3.detector == data.detector // 10))
    assert (np.all(data4.detector == data.detector ** 10))


def test_setters():
    """Tests setters working
    """

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

    with pytest.raises(ValueError):
        _test('h')
        _test('k')
        _test('l')
        _test('e')
        _test('temp')

    data = build_data()
    data.h = 3
    data.k = 3
    data.l = 3
    data.e = 3
    data.temp = 3
    data.monitor = np.ones(data.monitor.shape)
    data.time = np.ones(data.time.shape)
    data.data['detector'] = np.ones(data.intensity.shape)


def test_norms():
    """Test normalization types
    """
    data = build_data()
    data.m0 = 0
    assert (np.all(data.intensity == data.detector / data.monitor * data.m0))
    data.time_norm = True
    data.t0 = 0
    assert (np.all(data.intensity == data.detector / data.time * data.t0))


def test_error():
    """Tests exception handling
    """
    data = build_data()
    data._err = np.sqrt(data.detector)
    data.m0 = 0
    assert (np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))
    data._err = None
    assert (np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))
    del data._err
    assert (np.all(data.error == np.sqrt(data.detector) / data.monitor * data.m0))

    data.time_norm = True
    data._err = np.sqrt(data.detector)
    data.t0 = 0
    assert (np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))
    data._err = None
    assert (np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))
    del data._err
    assert (np.all(data.error == np.sqrt(data.detector) / data.time * data.t0))

    def _test():
        data.error = 10
        data.error = np.zeros(5)

    with pytest.raises(ValueError):
        _test()


def test_detailed_balance():
    """Test detailed balance factor
    """
    data = build_data()
    assert (np.all(data.detailed_balance_factor == 1. - np.exp(-data.e / BOLTZMANN_IN_MEV_K / data.temp)))


def test_scattering_function():
    """Test scattering function
    """
    from neutronpy import Material
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

    assert (np.all(data.scattering_function(material, 14.7) == 4 * np.pi / (
        material.total_scattering_cross_section) * ki / kf * data.detector))


def test_dynamic_susceptibility():
    """Test dynamic susceptibility
    """
    from neutronpy import Material
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

    assert (np.all(data.dynamic_susceptibility(material, 14.7) == 4 * np.pi / (
        material.total_scattering_cross_section) * ki / kf * data.detector * data.detailed_balance_factor))


def test_background_subtraction():
    """Test background subtraction
    """
    data = build_data(clean=True)
    background_data1 = build_data(clean=False)
    background_data2 = Data(detector=np.random.rand(101), monitor=np.full(101, 1, dtype=float),
                            time=np.full(101, 1, dtype=float), h=np.linspace(-1, 1, 101))
    try:
        data.subtract_background(background_data1, ret=False)
        data.subtract_background(background_data1, x='h', ret=False)
    except:
        pytest.fail('background subtraction failed')

    with pytest.raises(ValueError):
        data.subtract_background(background_data2, ret=False)


@patch("matplotlib.pyplot.show")
def test_plotting(mock_show):
    """Test plotting
    """
    data = build_data()
    with pytest.raises(AttributeError):
        data.plot()
    data.plot_default_x = 'h'
    data.plot_default_y = 'detector'
    data.plot()
    data.plot('h', 'intensity')
    data.plot('h', 'detector')
    data.plot('h', 'intensity', show_err=False)
    data.plot('h', 'intensity', output_file='plot_test.pdf')
    data.plot('h', 'intensity', smooth_options=dict(sigma=1))
    data.plot('h', 'intensity', fit_options=dict(function=functions.voigt, fixp=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 p=[20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2]))
    with pytest.raises(Exception):
        data.plot('h', 'intensity', fit_options=dict(function=functions.voigt, fixp=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                     p=[20., 0., 3., -0.15, 0.08, 0.2, 3., 0.15, 0.08, 0.2]))
    data.plot('h', 'intensity',
              to_bin=dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))
    data.plot('h', 'detector',
              to_bin=dict(h=[-1, 1., 41], k=[-0.1, 0.1, 1], l=[-0.1, 0.1, 1], e=[3.5, 4.5, 1], temp=[-300, 900, 1]))

    data3d = build_3d_data()
    data3d.plot(x='h', y='k', z='intensity')
    data3d.plot(x='h', y='k', z='intensity', to_bin=dict(h=[-1, 1, 41], k=[-1, 1, 41]))
    data3d.plot(x='h', y='k', z='detector', to_bin=dict(h=[-1, 1, 41], k=[-1, 1, 41]), smooth_options=dict(sigma=1),
                output_file='plot_test.pdf')
    with pytest.raises(KeyError):
        data3d.plot('h', 'k', 'blah')
        data3d.plot('h', 'k', 'w', 'blah')


if __name__ == '__main__':
    pytest.main()
