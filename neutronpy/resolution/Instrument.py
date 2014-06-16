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
    def __init__(self, name):
        self.name = name
        self.mono = _Monochromator('PG(002)', 25)
        self.ana = _Monochromator('PG(002)', 25)
        self.hcol = np.array([40, 40, 40, 40])
        self.vcol = np.array([120, 120, 120, 120])
        self.efixed = 14.7
        self.sample = _Sample(3.81, 3.81, 6.25, 90, 90, 90, 60)
        self.orient1 = np.array([1., 0., 0.])
        self.orient2 = np.array([0., 1., 0.])
