'''
Created on Jun 10, 2014

@author: davidfobes
'''
import numpy as np


class _Sample():
    def __init__(self, a, b, c, alpha, beta, gamma, mosaic):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mosaic = mosaic
        self.dir = 1


class _Monochromator():
    def __init__(self, tau, mosaic):
        self.tau = tau
        self.mosaic = mosaic
        self.dir = -1


class Instrument(object):
    '''An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    efixed : float
        Fixed energy, either ei or ef, depending on the instrument configuration.

    samp_abc : list(3)
        Sample lattice constants, [a, b, c], in Angstroms.

    samp_abg : list(3)
        Sample lattice parameters, [alpha, beta, gamma], in degrees.

    samp_mosaic : float
        Sample mosaic, in degrees.

    hcol : list(4)
        Horizontal Soller collimations in minutes of arc starting from the neutron guide.

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the neutron guide.

    mono_tau : string or float
        The monochromator reciprocal lattice vector in :math:`\AA^{-1}`,
        given either as a float, or as a string for common monochromator types.

    mono_mosaic : float
        The mosaic of the monochromator in minutes of arc.

    ana_tau : string or float
        The analyzer reciprocal lattice vector in :math:`\AA^{-1}`,
        given either as a float, or as a string for common analyzer types.

    ana_mosaic : float
        The mosaic of the monochromator in minutes of arc.

    orient1 : list(3)
        Miller indexes of the first reciprocal-space orienting vector for the sample coordinate system.

    orient2 : list(3)
        Miller indexes of the second reciprocal-space orienting vector for the sample coordinate system.

    Additional Parameters
    ---------------------
    The following parameters can be added later as needed for resolution calculation. The parameters
    listed above are the only parameters absolutely required for the Cooper-Nathans resolution
    calculation.

    '''
    def __init__(self, efixed=14.7, samp_abc=[3.81, 3.81, 6.25], samp_abg=[90, 90, 90], samp_mosaic=60,
                 hcol=[40, 40, 40, 40], vcol=[120, 120, 120, 120],
                 mono_tau='PG(002)', mono_mosaic=25,
                 ana_tau='PG(002)', ana_mosaic=25,
                 orient1=[1, 0, 0], orient2=[0, 1, 0]):
        [a, b, c] = samp_abc
        [alpha, beta, gamma] = samp_abg
        self.mono = _Monochromator(mono_tau, mono_mosaic)
        self.ana = _Monochromator(ana_tau, ana_mosaic)
        self.hcol = np.array(hcol)
        self.vcol = np.array(vcol)
        self.efixed = efixed
        self.sample = _Sample(a, b, c, alpha, beta, gamma, samp_mosaic)
        self.orient1 = np.array(orient1)
        self.orient2 = np.array(orient2)
