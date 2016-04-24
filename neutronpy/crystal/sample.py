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

    Attributes
    ----------
    a
    b
    c
    alpha
    beta
    gamma
    u
    v
    mosaic
    vmosaic
    direct
    width
    height
    depth
    shape
    astar
    bstar
    cstar
    alpha_rad
    beta_rad
    gamma_rad
    alphastar
    betastar
    gammastar
    alphastar_rad
    betastar_rad
    gammastar_rad
    abg_rad
    reciprocal_abc
    reciprocal_abg
    reciprocal_abg_rad
    lattice_type
    volume
    reciprocal_volume
    G
    Gstar
    Bmatrix

    Methods
    -------
    get_d_spacing
    get_q
    get_two_theta
    get_angle_between_planes

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
        r'''First orientation vector
        '''
        return self._u

    @u.setter
    def u(self, vec):
        self._u = np.array(vec)

    @property
    def v(self):
        r'''Second orientation vector
        '''
        return self._v

    @v.setter
    def v(self, vec):
        self._v = np.array(vec)
