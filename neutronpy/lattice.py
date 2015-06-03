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
    abg_rad
    lattice_type
    volume
    reciprocal_volume
    G
    Gstar
    
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
    def abg_rad(self):
        r'''Lattice angles in radians
        
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
                          [a * c * np.cos(beta), b * c * np.cos(alpha), c ** 2]])

    @property
    def Gstar(self):
        r'''Metric tensor of the reciprocal lattice
        
        '''

        return np.linalg.inv(self.G)

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

        return float(np.rad2deg(np.arccos(np.dot(np.dot(v1, self.Gstar), v2) /
                                          np.sqrt(np.dot(np.dot(v1, self.Gstar), v1)) /
                                          np.sqrt(np.dot(np.dot(v2, self.Gstar), v2)))))

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
        
        return 2 * np.rad2deg(np.arcsin(wavelength / 2 / self.get_d_spacing(hkl)))

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


if __name__ == "__main__":
    abc = [6.3496, 6.3496, 6.3496]
    abg = [90., 90., 90.]
    v1 = np.cross([1, 0, 0], [1, 1, 0])
    v2 = np.cross([2, 2, 1], [2, 2, -1])

    lattice = Lattice(abc, abg)
    print(v1, v2)

    print(lattice.volume)
    print(lattice.reciprocal_volume)
    print(lattice.get_angle_between_planes([1, 0, 0], [1, 1, 0]))
    print(lattice.get_two_theta([1, 1, 0], 2.35))
    print(lattice.get_d_spacing([1, 1, 0]))
    print(lattice.get_q([1, 1, 0]))
