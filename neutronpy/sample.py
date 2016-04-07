# -*- coding: utf-8 -*-
r'''Sample class for e.g. Instrument class

'''
import numpy as np
from .lattice import Lattice


class Sample(Lattice):
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

    u : array_like
        First orientation vector

    v : array_like
        Second orientation vector

    mosaic : float, optional
        Horizontal sample mosaic (FWHM) in arc minutes

    vmosaic : float, optional
        Vertical sample mosaic (FWHM) in arc minutes

    direct : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)

    width : float, optional
        Sample width in cm. Default: 1

    height : float, optional
        Sample height in cm. Default: 1

    depth : float, optional
        Sample thickness in cm. Default: 1

    shape : str, optional
        Sample shape type. Accepts 'rectangular' or 'cylindrical'.
        Default: 'rectangular'

    Returns
    -------
    Sample : object

    '''
    def __init__(self, a, b, c, alpha, beta, gamma, u=None, v=None, mosaic=None, vmosaic=None, direct=1,
                 width=None, height=None, depth=None, shape='rectangular'):
        super(Sample, self).__init__(a, b, c, alpha, beta, gamma)
        if u is not None:
            self._u = np.array(u)
        if v is not None:
            self._v = np.array(v)

        if mosaic is not None:
            self.mosaic = mosaic
        if vmosaic is not None:
            self.vmosaic = vmosaic
        self.dir = direct

        self.shape_type = shape
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if depth is not None:
            self.depth = depth

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
