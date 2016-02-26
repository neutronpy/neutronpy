import numpy as np


class Lattice(object):
    u'''Class to describe a generic lattice system defined by lattice six
    parameters, (three constants and three angles).

    Parameters
    ----------
    abc : array_like
        List of lattice constants *a*, *b*, and *c* in \u212B

    abg : array_like
        List of lattice angles \U0001D6FC, \U0001D6FD, and \U0001D6FE in
        degrees

    Returns
    -------
    lattice : object
        Object containing lattice information

    Attributes
    ----------
    a
    b
    c
    astar
    bstar
    cstar
    alpha
    beta
    gamma
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

    def __init__(self, abc, abg):
        self.abc = np.array(abc)
        self.abg = np.array(abg)

    @property
    def a(self):
        r'''First lattice constant in Angstrom
        '''
        return self.abc[0]

    @property
    def b(self):
        r'''Second lattice constant in Angstrom

        '''
        return self.abc[1]

    @property
    def c(self):
        r'''Third lattice constant in Angstrom
        '''
        return self.abc[2]

    @property
    def alpha(self):
        r'''First lattice angle in degrees
        '''
        return self.abg[0]

    @property
    def beta(self):
        r'''Second lattice angle in degrees
        '''
        return self.abg[1]

    @property
    def gamma(self):
        r'''Third lattice angle in degrees
        '''
        return self.abg[2]

    @property
    def alpha_rad(self):
        r'''First lattice angle in radians
        '''
        return self.abg_rad[0]

    @property
    def beta_rad(self):
        r'''Second lattice angle in radians
        '''
        return self.abg_rad[1]

    @property
    def gamma_rad(self):
        r'''Third lattice angle in radians
        '''
        return self.abg_rad[2]

    @property
    def astar(self):
        r'''First inverse lattice constant in inverse Angstrom
        '''
        return self.b * self.c * np.sin(self.alpha_rad) / self.volume

    @property
    def bstar(self):
        r'''Second inverse lattice constant in inverse Angstrom
        '''
        return self.a * self.c * np.sin(self.beta_rad) / self.volume

    @property
    def cstar(self):
        r'''Third inverse lattice constant in inverse Angstrom
        '''
        return self.a * self.b * np.sin(self.gamma) / self.volume

    @property
    def alphastar_rad(self):
        r'''First inverse lattice angle in radians
        '''
        return np.arccos((np.cos(self.beta_rad) * np.cos(self.gamma_rad) -
                          np.cos(self.alpha_rad)) /
                         (np.sin(self.beta_rad) * np.sin(self.gamma_rad)))

    @property
    def betastar_rad(self):
        r'''Second inverse lattice angle in radians
        '''
        return np.arccos((np.cos(self.alpha_rad) *
                          np.cos(self.gamma_rad) -
                          np.cos(self.beta_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.gamma_rad)))

    @property
    def gammastar_rad(self):
        r'''Third inverse lattice angle in radians
        '''
        return np.arccos((np.cos(self.alpha_rad) * np.cos(self.beta_rad) -
                          np.cos(self.gamma_rad)) /
                         (np.sin(self.alpha_rad) * np.sin(self.beta_rad)))

    @property
    def alphastar(self):
        r'''First inverse lattice angle in degrees
        '''
        return np.rad2deg(self.alphastar_rad)

    @property
    def betastar(self):
        r'''First inverse lattice angle in degrees
        '''
        return np.rad2deg(self.betastar_rad)

    @property
    def gammastar(self):
        r'''First inverse lattice angle in degrees
        '''
        return np.rad2deg(self.gammastar_rad)

    @property
    def reciprocal_abc(self):
        r'''Reciprocal lattice constants in inverse Angstrom returned in list
        '''
        return [self.astar, self.bstar, self.cstar]

    @property
    def reciprocal_abg(self):
        r'''Reciprocal lattice angles in degrees returned in list
        '''
        return [self.alphastar, self.betastar, self.gammastar]

    @property
    def reciprocal_abg_rad(self):
        r'''Reciprocal lattice angles in radians returned in list
        '''
        return [self.alphastar_rad, self.betastar_rad, self.gammastar_rad]

    @property
    def abg_rad(self):
        r'''Lattice angles in radians returned in list
        '''
        return np.deg2rad(self.abg)

    @property
    def lattice_type(self):
        r'''Type of lattice determined by the provided lattice constants and angles

        '''

        if len(np.unique(self.abc)) == 3 and len(np.unique(self.abg)) == 3:
            return 'triclinic'
        elif len(np.unique(self.abc)) == 3 and self.abg[1] != 90 and np.all(self.abg[:3:2] == 90):
            return 'monoclinic'
        elif len(np.unique(self.abc)) == 3 and np.all(self.abg == 90):
            return 'orthorhombic'
        elif len(np.unique(self.abc)) == 1 and len(np.unique(self.abg)) == 1 and np.all(self.abg < 120) and np.all(self.abg != 90):
            return 'rhombohedral'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(self.abg == 90):
            return 'tetragonal'
        elif len(np.unique(self.abc)) == 2 and self.abc[0] == self.abc[1] and np.all(self.abg[0:2] == 90) and self.abg[2] == 120:
            return 'hexagonal'
        elif len(np.unique(self.abc)) == 1 and np.all(self.abg == 90):
            return 'cubic'
        else:
            raise ValueError('Provided lattice constants and angles do not resolve to a valid Bravais lattice')

    @property
    def volume(self):
        u'''Volume of the unit cell in \u212B\ :sup:`3`

        '''
        return np.sqrt(np.linalg.det(self.G))

    @property
    def reciprocal_volume(self):
        u'''Volume of the reciprocal unit cell in (\u212B\ :sup:`-1`\ )\ :sup:`3`

        '''
        return np.sqrt(np.linalg.det(self.Gstar))

    @property
    def G(self):
        r'''Metric tensor of the real space lattice

        '''

        a, b, c = self.abc
        alpha, beta, gamma = self.abg_rad

        return np.matrix([[a ** 2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                          [a * b * np.cos(gamma), b ** 2, b * c * np.cos(alpha)],
                          [a * c * np.cos(beta), b * c * np.cos(alpha), c ** 2]], dtype=float)

    @property
    def Gstar(self):
        r'''Metric tensor of the reciprocal lattice

        '''

        return np.linalg.inv(self.G)

    @property
    def Bmatrix(self):
        r'''Cartesian basis matrix in reciprocal units such that
        Bmatrix*Bmatrix.T = Gstar

        '''

        return np.matrix([[self.astar, self.bstar * np.cos(self.gammastar_rad), self.cstar * np.cos(self.betastar_rad)],
                          [0, self.bstar * np.sin(self.gammastar_rad), - self.cstar * np.sin(self.betastar_rad) * np.cos(self.alpha_rad)],
                          [0, 0, 1. / self.c]], dtype=float)

    def get_d_spacing(self, hkl):
        u'''Returns the d-spacing of a given reciprocal lattice vector.

        Parameters
        ----------
        hkl : array_like
            Reciprocal lattice vector in r.l.u.

        Returns
        -------
        d : float
            The d-spacing in \u212B

        '''
        hkl = np.array(hkl)

        return float(1 / np.sqrt(np.dot(np.dot(hkl, self.Gstar), hkl)))

    def get_angle_between_planes(self, v1, v2):
        r'''Returns the angle :math:`\phi` between two reciprocal lattice
        vectors (or planes as defined by the vectors normal to the plane).

        Parameters
        ----------
        v1 : array_like
            First reciprocal lattice vector in units r.l.u.

        v2 : array_like
            Second reciprocal lattice vector in units r.l.u.

        Returns
        -------
        phi : float
            The angle between v1 and v2 in degrees

        '''

        v1, v2 = np.array(v1), np.array(v2)

        return float(np.rad2deg(np.arccos(np.inner(np.inner(v1, self.Gstar), v2) /
                                          np.sqrt(np.inner(np.inner(v1, self.Gstar), v1)) /
                                          np.sqrt(np.inner(np.inner(v2, self.Gstar), v2)))))

    def get_two_theta(self, hkl, wavelength):
        u'''Returns the detector angle 2\U0001D703 for a given reciprocal
        lattice vector and incident wavelength.

        Parameters
        ----------
        hkl : array_like
            Reciprocal lattice vector in r.l.u.

        wavelength : float
            Wavelength of the incident beam in \u212B

        Returns
        -------
        two_theta : float
            The angle of the detector 2\U0001D703 in degrees

        '''

        return 2 * np.rad2deg(np.arcsin(wavelength / 2 /
                                        self.get_d_spacing(hkl)))

    def get_q(self, hkl):
        u'''Returns the magnitude of *Q* for a given reciprocal lattice
        vector in \u212B\ :sup:`-1`.

        Parameters
        ----------
        hkl : array_like
            Reciprocal lattice vector in r.l.u.

        Returns
        -------
        q : float
            The magnitude of the reciprocal lattice vector *Q* in
            \u212B\ :sup:`-1`

        '''

        return 2 * np.pi / self.get_d_spacing(hkl)


class Goniometer(object):
    r'''Defines a goniometer

    '''

    def __init__(self, u, theta_u, v, theta_v, sgu, sgl, omega=0):
        self.u = u
        self.theta_u = theta_u

        self.v = v
        self.theta_v = theta_v

        self.sgu = sgu
        self.sgl = sgl

        self.omega = 0

    @property
    def omega_rad(self):
        return self.omega_rad

    @property
    def sgu_rad(self):
        return np.deg2rad(self.sgu)

    @property
    def sgl_rad(self):
        return np.deg2rad(self.sgl)

    @property
    def theta_rad(self):
        return np.arctan((self.ki - self.kf * np.cos(self.phi)) / (self.kf * np.sin(self.phi)))

    @property
    def theta(self):
        pass

    @property
    def N(self):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(self.sgu_rad), -np.sin(self.sgu_rad)],
                          [0, np.sin(self.sgu_rad), np.cos(self.sgu_rad)]])

    @property
    def M(self):
        return np.matrix([[np.cos(self.sgl_rad), 0, np.sin(self.sgl_rad)],
                          [0, 1, 0],
                          [-np.sin(self.sgl_rad), 0, np.cos(self.sgl_rad)]])

    @property
    def Omega(self):
        return np.matrix([[np.cos(self.omega_rad), -np.sin(self.omega_rad), 0],
                          [np.sin(self.omega_rad), np.cos(self.omega_rad), 0],
                          [0, 0, 1]])

    @property
    def Theta(self):
        return np.matrix([[np.cos(self.theta_rad), -np.sin(self.theta_rad), 0],
                          [np.sin(self.theta_rad), np.cos(self.theta_rad), 0],
                          [0, 0, 1]])

    @property
    def T_c(self):
        return np.matrix([self.u, self.v, np.cross(self.u, self.v)]).T

    @property
    def T_phi(self):
        return np.matrix([self.u_phi(np.deg2rad(self.theta_u), self.sgl_rad, self.sgu_rad),
                          self.u_phi(np.deg2rad(self.theta_v), self.sgl_rad, self.sgu_rad),
                          self.u_phi(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))]).T

    @property
    def R(self):
        return self.Omega * self.M * self.N

    @property
    def U(self):
        r'''Defines an orientation matrix based on supplied goniometer angles

        '''
        return self.T_phi * np.linalg.inv(self.T_c)

    def u_phi(self, omega, chi, phi):
        return [np.cos(omega) * np.cos(chi) * np.cos(phi) - np.sin(omega) * np.sin(phi),
                np.cos(omega) * np.cos(chi) * np.sin(phi) + np.sin(omega) * np.cos(phi),
                np.cos(omega) * np.sin(chi)]
