# -*- coding: utf-8 -*-
r"""Sample class for e.g. Instrument class

"""
import numpy as np

from .lattice import Lattice
from .tools import gram_schmidt


class Sample(Lattice):
    u"""Private class containing sample information.

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

    distance : float, optional
        Distance from source (used for Time of Flight resolution
        calculations). Default: None

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

    """

    def __init__(self, a, b, c, alpha, beta, gamma, u=None, v=None, mosaic=None, vmosaic=None, direct=1,
                 width=None, height=None, depth=None, shape='rectangular', distance=None):
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
        if distance is not None:
            self.distance = distance

    def __repr__(self):
        args = ', '.join([str(getattr(self, key)) for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']])
        kwargs = ', '.join(['{0}={1}'.format(key, getattr(self, key)) for key in
                            ['u', 'v', 'mosaic', 'vmosaic', 'direct', 'width', 'height', 'depth', 'shape'] if
                            getattr(self, key, None) is not None])
        return "Sample({0}, {1})".format(args, kwargs)

    def __eq__(self, right):
        self_parent_keys = sorted(list(self.__dict__.keys()))
        right_parent_keys = sorted(list(right.__dict__.keys()))

        if not np.all(self_parent_keys == right_parent_keys):
            return False

        for key, value in self.__dict__.items():
            right_parent_val = getattr(right, key)
            if not np.all(value == right_parent_val):
                print(value, right_parent_val)
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    @property
    def u(self):
        r"""First orientation vector
        """
        return self._u

    @u.setter
    def u(self, vec):
        self._u = np.array(vec)

    @property
    def v(self):
        r"""Second orientation vector
        """
        return self._v

    @v.setter
    def v(self, vec):
        self._v = np.array(vec)

    @property
    def direct(self):
        return self.dir

    @direct.setter
    def direct(self, value):
        self.dir = value

    @property
    def Umatrix(self):
        u"""
        """
        ortho_basis = gram_schmidt(np.vstack((self.u, self.v)))

        return np.vstack((ortho_basis, np.cross(ortho_basis[0], ortho_basis[1])))

    @property
    def UBmatrix(self):
        u"""
        """
        return self.Umatrix * self.Bmatrix

    def get_phi(self, Q):
        u"""Get out-of-plane scattering angle.

        Parameters
        ----------
        hkl: array_like

        wavelength : float

        Returns
        -------
        phi : float
            The out-of-plane angle

        """
        return self.get_angle_between_planes(Q, np.cross(self.u, self.v))
