# -*- coding: utf-8 -*-
r'''Testing of the resolution library

'''
import os
from copy import deepcopy
import unittest
from mock import patch
from matplotlib import use
import numpy as np
from neutronpy import instrument
from neutronpy.io import load_instrument
from neutronpy.sample import Sample
use('Agg')


def angle2(x, y, z, h, k, l, lattice):
    r'''Function necessary for Prefactor functions
    '''
    latticestar = instrument._star(lattice)[-1]

    return np.arccos(2 * np.pi * (h * x + k * y + l * z) / instrument._modvec([x, y, z], lattice) / instrument._modvec([h, k, l], latticestar))


def SqwDemo(H, K, L, W, p):
    r'''Example Scattering function for convolution tests
    '''
    del K, L
    Deltax = p[0]
    Deltay = p[1]
    Deltaz = p[2]
    cc = p[3]
    Gamma = p[4]

    omegax = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltax ** 2)
    omegay = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltay ** 2)
    omegaz = np.sqrt(cc ** 2 * (np.sin(2 * np.pi * H)) ** 2 + Deltaz ** 2)

    lorx = 1 / np.pi * Gamma / ((W - omegax) ** 2 + Gamma ** 2)
    lory = 1 / np.pi * Gamma / ((W - omegay) ** 2 + Gamma ** 2)
    lorz = 1 / np.pi * Gamma / ((W - omegaz) ** 2 + Gamma ** 2)

    sqw0 = lorx * (1 - np.cos(np.pi * H)) / omegax / 2
    sqw1 = lory * (1 - np.cos(np.pi * H)) / omegay / 2
    sqw2 = lorz * (1 - np.cos(np.pi * H)) / omegaz / 2

    sqw = np.vstack((sqw0, sqw1, sqw2))

    return sqw


def SMADemo(H, K, L, p):
    r'''Example Scattering function for convolution tests
    '''
    del K, L
    Deltax = p[0]
    Deltay = p[1]
    Deltaz = p[2]
    cc = p[3]
    Gamma = p[4]

    omegax = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltax ** 2)
    omegay = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltay ** 2)
    omegaz = np.sqrt(cc ** 2 * (np.sin(2. * np.pi * H.flatten())) ** 2 + Deltaz ** 2)
    w0 = np.vstack((omegax, omegay, omegaz))

    S = np.vstack(((1. - np.cos(np.pi * H.flatten())) / omegax / 2.,
                   (1. - np.cos(np.pi * H.flatten())) / omegay / 2.,
                   (1. - np.cos(np.pi * H.flatten())) / omegaz / 2.))

    HWHM = np.ones(S.shape) * Gamma

    return [w0, S, HWHM]


def PrefDemo(H, K, L, W, EXP, p):
    r'''Prefactor example for convolution tests
    '''
    [sample, rsample] = EXP.get_lattice()

    q2 = instrument._modvec([H, K, L], rsample) ** 2

    sd = q2 / (16 * np.pi ** 2)
    ff = 0.0163 * np.exp(-35.883 * sd) + 0.3916 * np.exp(-13.223 * sd) + 0.6052 * np.exp(-4.339 * sd) - 0.0133

    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.0 * polx * p[5]
    prefactor[1, :] = ff ** 2.0 * poly * p[5]
    prefactor[2, :] = ff ** 2.0 * polz * p[5]

    bgr = np.ones(H.shape) * p[6]

    return [prefactor, bgr]


def PrefDemo2(H, K, L, W, EXP, p):
    r'''Prefactor example for convolution tests

    No background

    '''
    [sample, rsample] = EXP.get_lattice()

    q2 = instrument._modvec([H, K, L], rsample) ** 2

    sd = q2 / (16 * np.pi ** 2)
    ff = 0.0163 * np.exp(-35.883 * sd) + 0.3916 * np.exp(-13.223 * sd) + 0.6052 * np.exp(-4.339 * sd) - 0.0133

    alphax = angle2(1, 0, 0, H, K, L, sample)
    alphay = angle2(0, 1, 0, H, K, L, sample)
    alphaz = angle2(0, 0, 1, H, K, L, sample)

    polx = np.sin(alphax) ** 2
    poly = np.sin(alphay) ** 2
    polz = np.sin(alphaz) ** 2

    prefactor = np.zeros((3, len(H)))
    prefactor[0, :] = ff ** 2.0 * polx * p[5]
    prefactor[1, :] = ff ** 2.0 * poly * p[5]
    prefactor[2, :] = ff ** 2.0 * polz * p[5]

    return prefactor


def PrefDemo3(H, K, L, W, EXP, p):
    r'''Prefactor example for convolution tests

    No prefactor

    '''

    return


class ResolutionTest(unittest.TestCase):
    r'''Unit tests for Resolution calculations
    '''
    def __init__(self, *args, **kwargs):
        r'''Setup basic instrument
        '''
        super(ResolutionTest, self).__init__(*args, **kwargs)

        self.sumIavg = 1654.37911333
        self.sumIstd = 0.5

        instr = instrument.Instrument(test=1)

        instr.method = 0
        instr.mono.tau = 'PG(002)'
        instr.mono.mosaic = 25
        instr.ana.tau = 'PG(002)'
        instr.ana.mosaic = 25
        instr.sample.a = 6
        instr.sample.b = 7
        instr.sample.c = 8
        instr.sample.alpha = 90
        instr.sample.beta = 90
        instr.sample.gamma = 90
        instr.hcol = [40, 40, 40, 40]
        instr.vcol = [120, 120, 120, 120]
        instr.efixed = 14.7
        instr.orient1 = np.array([1, 0, 0])
        instr.orient2 = np.array([0, 1, 0])

        self.EXP_coopernathans = deepcopy(instr)
        instr.method = 1
        self.EXP_popovici = deepcopy(instr)

    def test_cooper_nathans(self):
        '''Test Cooper Nathans method
        '''
        R0 = 2117.45739160280
        RMS = np.array([[9154.39386475516, 7.32203491574463e-11, 0, 7.11894676107400e-12],
                        [2.68712790277282e-10, 340628.383580632, 0, -32536.7077302429],
                        [0, 0, 634.724632931705, 0],
                        [2.58004722905037e-11, -32536.7077302429, 0, 3114.58144514260]])
        ResVol0 = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(RMS)) * 2
        angles0 = np.array([-20.58848852, -41.17697704, -78.6627354, 22.67452921, -20.58848852, -41.17697704])
        BraggWidths0 = np.array([0.0492235489748347, 0.00806951257792662, 0.186936902874783, 1.82137589975272, 0.0843893950600324])

        EXP = self.EXP_coopernathans
        hkle = [1., 0., 0., 0.]

        EXP.calc_resolution(hkle)

        NP = EXP.RMS
        R = EXP.R0
        BraggWidths = instrument.get_bragg_widths(NP)
        angles = self.EXP_popovici.get_angles_and_Q(hkle)[0]
        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP)) * 2

        self.assertTrue(np.all(np.abs((RMS - NP)) < 100))
        self.assertAlmostEqual(R, R0, 3)
        self.assertAlmostEqual(ResVol, ResVol0, 5)
        self.assertTrue(np.all(np.abs((BraggWidths - BraggWidths0)) < 0.1))
        self.assertTrue(np.all(np.abs((angles0 - angles)) < 0.1))

    def test_popovici(self):
        '''Test Popovici method
        '''
        R0 = 2117.46377630698
        RMS = np.array([[9154.44276618996, 4.78869185251432e-08, 0, 4.57431754676102e-09],
                        [8.53192164855333e-08, 340633.245599205, 0, -32537.1653207760],
                        [0, 0, 634.821032587120, 0],
                        [8.14983128960581e-09, -32537.1653207760, 0, 3114.62458263531]])
        ResVol0 = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(RMS)) * 2
        angles0 = np.array([-20.58848852, -41.17697704, -78.6627354, 22.67452921, -20.58848852, -41.17697704])
        BraggWidths0 = np.array([0.0492234175028573, 0.00806945498774637, 0.186922708845071, 1.82136489553849, 0.0843888106622307])

        EXP = self.EXP_popovici
        hkle = [1, 0, 0, 0]

        EXP.calc_resolution(hkle)

        NP = self.EXP_popovici.RMS
        R = self.EXP_popovici.R0
        BraggWidths = instrument.get_bragg_widths(NP)
        angles = self.EXP_popovici.get_angles_and_Q(hkle)[0]

        ResVol = (2 * np.pi) ** 2 / np.sqrt(np.linalg.det(NP)) * 2

        self.assertTrue(np.all(np.abs((RMS - NP) / 1e4) < 0.1))
        self.assertAlmostEqual(R, R0, 3)
        self.assertAlmostEqual(ResVol, ResVol0, 5)
        self.assertTrue(np.all(np.abs((BraggWidths - BraggWidths0)) < 0.1))
        self.assertTrue(np.all(np.abs((angles0 - angles)) < 1e-3))

    def test_4d_conv(self):
        '''Test 4d convolution
        '''
        sample = Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = instrument.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')
        EXP.moncor = 0

        p = np.array([3, 3, 3, 30, 0.4, 6e4, 40])
        H1, K1, L1, W1 = 1.5, 0, 0.35, np.arange(20, -0.5, -0.5)

        I11 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'fix', [5, 0], p)
        I12 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'fix', [15, 0], p)
        I13 = EXP.resolution_convolution(SqwDemo, PrefDemo, 2, (H1, K1, L1, W1), 'mc', None, p, 13)

        sumI11, sumI12, sumI13 = np.sum(I11), np.sum(I12), np.sum(I13)

#         self.assertTrue(np.abs(self.sumIavg - sumI11) < self.sumIstd)
#         self.assertTrue(np.abs(self.sumIavg - sumI12) < self.sumIstd)
#         self.assertTrue(np.abs(self.sumIavg - sumI13) < self.sumIstd)

        EXP.resolution_convolution(SqwDemo, PrefDemo2, 1, (H1, K1, L1, W1), 'fix', None, p)
        self.assertRaises(ValueError, EXP.resolution_convolution, SqwDemo, PrefDemo3, 0, (H1, K1, L1, W1), 'fix', [5, 0], p)

    def test_sma_conv(self):
        '''Test SMA convolution
        '''
        sample = Sample(6, 7, 8, 90, 90, 90)
        sample.u = [1, 0, 0]
        sample.v = [0, 0, 1]
        EXP = instrument.Instrument(14.7, sample, hcol=[80, 40, 40, 80], vcol=[120, 120, 120, 120], mono='pg(002)', ana='pg(002)')
        EXP.moncor = 0

        p = np.array([3, 3, 3, 30, 0.4, 6e4, 40])
        H1, K1, L1, W1 = 1.5, 0, 0.35, np.arange(20, -0.5, -0.5)

        I14 = EXP.resolution_convolution_SMA(SMADemo, PrefDemo, 2, (H1, K1, L1, W1), 'fix', [15, 0], p)
        I15 = EXP.resolution_convolution_SMA(SMADemo, PrefDemo, 2, (H1, K1, L1, W1), 'mc', [1], p, 13)

        sumI14, sumI15 = np.sum(I14), np.sum(I15)

#         self.assertTrue(np.abs(self.sumIavg - sumI14) < self.sumIstd)
#         self.assertTrue(np.abs(self.sumIavg - sumI15) < self.sumIstd)

        EXP.resolution_convolution_SMA(SMADemo, PrefDemo2, 1, (H1, K1, L1, W1), 'fix', None, p)
        self.assertRaises(ValueError, EXP.resolution_convolution_SMA, SMADemo, PrefDemo3, 0, (H1, K1, L1, W1), 'fix', None, p)

    @patch("matplotlib.pyplot.show")
    def test_plotting(self, mock_show):
        '''Test Plotting methods
        '''
        EXP = instrument.Instrument()
        EXP.plot_instrument([1, 0, 0, 0])
        EXP.plot_projections([1, 0, 0, 0])
        EXP.calc_resolution([[1, 2], 0, 0, 0])
        EXP.plot_projections([[1, 2], 0, 0, 0])

        EXP.beam.width = 1
        EXP.beam.height = 1
        EXP.mono.width = 1
        EXP.mono.height = 1
        EXP.sample.width = 1
        EXP.sample.height = 1
        EXP.sample.depth = 1
        EXP.ana.width = 1
        EXP.ana.height = 1
        EXP.detector.width = 1
        EXP.detector.height = 1
        EXP.arms = [10, 10, 10, 10]
        EXP.plot_instrument([1, 0, 0, 0])

    def test_load_par_cfg(self):
        '''Test instrument loading using par/cfg files
        '''
        parfile = os.path.join(os.path.dirname(__file__), 'test.par')
        cfgfile = os.path.join(os.path.dirname(__file__), 'test.cfg')
        load_instrument(parfile, cfgfile)

        parfile = os.path.join(os.path.dirname(__file__), 'test2.par')
        cfgfile = os.path.join(os.path.dirname(__file__), 'test2.cfg')
        load_instrument(parfile, cfgfile)

    def test_sample(self):
        '''Test Sample class
        '''
        sample = Sample(1, 1, 1, 90, 90, 90, mosaic=60, direct=-1, u=[1, 0, 0], v=[0, 1, 0])
        self.assertTrue(isinstance(sample.u, np.ndarray))
        self.assertTrue(isinstance(sample.v, np.ndarray))

    def test_GetTau(self):
        '''Test monochromator crystal tau value finder
        '''
        self.assertTrue(instrument.GetTau(1.87325, getlabel=True) == 'pg(002)')
        self.assertTrue(instrument.GetTau(1.8, getlabel=True) == '')
        self.assertTrue(instrument.GetTau(10) == 10)
        self.assertRaises(KeyError, instrument.GetTau, 'blah')

    def test_CleanArgs_err(self):
        '''Test exception capture in CleanArgs
        '''
        pass

    def test_fproject(self):
        '''Test projection function
        '''
        x = np.ones((4, 4, 1))
        instrument.fproject(x, 0)
        instrument.fproject(x, 1)
        instrument.fproject(x, 2)

    def test_constants(self):
        '''Test constants
        '''
        self.EXP_popovici.moncor = 0
        self.assertTrue(self.EXP_popovici.moncor == 0)

    def test_errors(self):
        '''Test exception handling
        '''
        EXP = instrument.Instrument()
        EXP.sample.u = [1, 0, 0]
        EXP.sample.v = [2, 0, 0]
        self.assertRaises(ValueError, EXP.calc_resolution, [1, 1, 0, 0])

    def test_calc_res_cases(self):
        '''Test different resolution cases
        '''
        EXP = instrument.Instrument()
        EXP.sample.shape = np.eye(3)
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.sample.shape = np.eye(3)[np.newaxis].reshape((1, 3, 3))
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.horifoc = 1
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.moncor = 1
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.method = 1
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.ana.thickness = 1
        EXP.ana.Q = 1.5
        EXP.calc_resolution([1, 0, 0, 0])

        EXP.Smooth = instrument._dummy()
        EXP.Smooth.X = 1
        EXP.Smooth.Y = 1
        EXP.Smooth.Z = 1
        EXP.Smooth.E = 1
        EXP.calc_resolution([1, 0, 0, 0])

    def test_projection_calc(self):
        '''Test different cases of resolution ellipse slices/projections
        '''
        EXP = instrument.Instrument()
        EXP.calc_resolution([1, 0, 0, 0])
        EXP.calc_projections([0, 1, 0, 0])
        EXP.get_resolution_params([0, 1, 0, 0], 'QxQy', 'slice')
        self.assertRaises(ValueError, EXP.get_resolution_params, [1, 1, 0, 0], 'QxQy', 'slice')

        EXP = instrument.Instrument()
        EXP.get_resolution_params([1, 0, 0, 0], 'QxQy', 'slice')
        EXP.get_resolution_params([1, 0, 0, 0], 'QxQy', 'project')
        EXP.get_resolution_params([1, 0, 0, 0], 'QxW', 'slice')
        EXP.get_resolution_params([1, 0, 0, 0], 'QxW', 'project')
        EXP.get_resolution_params([1, 0, 0, 0], 'QyW', 'slice')
        EXP.get_resolution_params([1, 0, 0, 0], 'QyW', 'project')


if __name__ == '__main__':
    unittest.main()
