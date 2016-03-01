# -*- coding: utf-8 -*-
r'''Sample class for e.g. Instrument class

'''
import numpy as np


class Sample(object):
    u'''Private class containing sample information.

    Parameters
    ----------
    a : float
        Unit cell length in angstroms

    b : float
        Unit cell length in angstroms

    c : float
        Unit cell length in angstroms

    alpha : float
        Angle between b and c in degrees

    beta : float
        Angle between a and c in degrees

    gamma : float
        Angle between a and b in degrees

    mosaic : float, optional
        Horizontal sample mosaic (FWHM) in arc minutes

    vmosaic : float, optional
        Vertical sample mosaic (FWHM) in arc minutes

    direct : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)

    u : array_like
        First orientation vector

    v : array_like
        Second orientation vector

    Returns
    -------
    Sample : object

    '''
    def __init__(self, a, b, c, alpha, beta, gamma, mosaic=None, vmosaic=None, direct=1, u=None, v=None):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if mosaic is not None:
            self.mosaic = mosaic
        if vmosaic is not None:
            self.vmosaic = vmosaic
        self.dir = direct
        if u is not None:
            self._u = np.array(u)
        if v is not None:
            self._v = np.array(v)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, vec):
        self._u = np.array(vec)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, vec):
        self._v = np.array(vec)