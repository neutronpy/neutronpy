import neutronpy.constants as const
import numpy as np
from numbers import Number


class _Atom(object):
    r'''Class for adding atoms to the Material class.

    Parameters
    ----------
    ion : string
        The name of the Atom, or ion if necessary
    pos : list(3)
        The position of the Atom in the chosen geometry
    dpos : list(3), optional
        Deviations from the position pos
    occupancy: float, optional
        Occupancy of the _Atom (*e.g.* if there is partial occupancy from doping)
    Mcell : float, optional
        The mass of the unit cell. If assigned, normalize scattering lengths to the
        square-root of the mass of the atom

    Returns
    -------
    output : object
        Atom object defining an individual atom in a unit cell of a single crystal

    '''
    def __init__(self, ion, pos, dpos=None, occupancy=1., Mcell=None, massNorm=False):
        self.ion = ion
        self.pos = pos
        if dpos is None:
            self.dpos = [0., 0., 0.]
        else:
            self.dpos = dpos
        self.occupancy = occupancy
        self.Mcell = Mcell

        if massNorm is True:
            self.mass = const.periodic_table()[ion]['mass']
            self.b = (const.scattering_lengths()[ion]['Coh b']
                      * self.occupancy
                      * self.Mcell
                      / np.sqrt(self.mass))
        else:
            self.b = const.scattering_lengths()[ion]['Coh b'] * self.occupancy


class Material(object):
    r'''Class for the Material being supplied for the structure factor calculation

    Parameters
    ----------
    data : dictionary
        data is a dictionary containing all of the atoms
        and their positions, with optional occupancy
        and variances in positions (dpos), which may be
        used for Debye-Waller factor. This dictionary has
        the format:

        .. code-block:: python

            {'name': str,
             'composition': [{'ion': str,
                              'pos': [float, float, float],
                              'dpos': [float, float, float],
                              'occupancy': float}],
             'debye-waller': bool,
             'massNorm': bool,
             'formulaUnits': float,
             'lattice': [float, float, float]}

    Returns
    -------
    output : object
        Material Object defining a single crystal.
        
    Methods
    -------
    calc_str_fac
    

    '''

    def __init__(self, crystal):
        if 'formulaUnits' not in crystal:
            crystal['formulaUnits'] = 1.

        self.muCell = 0.
        for value in crystal['composition']:
            if 'occupancy' not in value:
                value['occupancy'] = 1.
            self.muCell += const.periodic_table()[value['ion']]['mass'] * value['occupancy']

        self.Mcell = crystal['formulaUnits'] * self.muCell

        if 'lattice' in crystal:
            self.abc = crystal['lattice']
            self.abcs = np.array([2. * np.pi / a for a in self.abc])

        self.atoms = []
        for value in crystal['composition']:
            if 'dpos' not in value:
                value['dpos'] = np.zeros(3)
            if crystal['debye-waller'] and np.all(value['dpos'] == np.zeros(3)) and self.abc:
                # If debye-waller should be included, but no dpos is provided, use value
                # for bulk copper (1/17.38) / abc (in \AA^-1)
                value['dpos'] = np.array([1 / 17.38] * 3) / self.abc
            if 'occupancy' not in value:
                value['occupancy'] = 1.
            self.atoms.append(_Atom(value['ion'],
                                    value['pos'],
                                    value['dpos'],
                                    value['occupancy'],
                                    self.Mcell,
                                    crystal['massNorm']))

    def calc_str_fac(self, hkl):
        r'''Calculates the structural form factor of the material.

        Parameters
        ----------
        hkl : tuple of floats, or tuple of ndarrays
            Reciprocal lattice positions at which the structure
            factor should be calculated

        Returns
        -------
         FQ : float or ndarray
             Structure factor at the position or positions specified

        '''

        h, k, l = hkl

        # Determines shape of input variables to build FQ = 0 array
        dims = []
        for x in hkl:
            if isinstance(x, np.ndarray):
                dims.append(x.shape)
            elif isinstance(x, (list, tuple)):
                dims.append((len(x),))
            elif isinstance(x, Number):
                dims.append(1)
        
        dims_str = np.array([str(x) for x in dims], dtype=str)

        if np.unique(dims_str).size == 1:
            FQ = np.zeros(dims[0])
        elif np.unique(dims_str[np.where([isinstance(x, tuple) for x in dims])]).size == 1:
            FQ = np.zeros(dims[np.where([isinstance(x, tuple) for x in dims])[0][0]])
        else:
            raise ValueError("Dimensions of 'hkl' elements are not compatible. An " \
                             "element must be either the same shape as an other " \
                             "non-decimal element, or a decimal number.")

        # Ensures input arrays are complex ndarrays
        if isinstance(h, (np.ndarray, list, tuple)):
            h = np.array(h).astype(complex, casting='unsafe')
        if isinstance(k, (np.ndarray, list, tuple)):
            k = np.array(k).astype(complex, casting='unsafe')
        if isinstance(l, (np.ndarray, list, tuple)):
            l = np.array(l).astype(complex, casting='unsafe')
            
        # construct structure factor
        for atom in self.atoms:
            FQ += atom.b * np.exp(1j * 2. * np.pi * (h * atom.pos[0] + k * atom.pos[1] + l * atom.pos[2])) * \
                  np.exp(-(2. * np.pi * (h * atom.dpos[0] + k * atom.dpos[1] + l * atom.dpos[2])) ** 2)

        return FQ


class Ion(object):
    r'''Class defining a magnetic ion.

    Parameters
    ----------
    ion : str
        Name of the atom, ion or anion, *e.g.* 'Fe2+'.

    Returns
    -------
    output : Object
        Ion object defining a single magnetic ion.

    Methods
    -------
    calc_mag_form_fac
    '''

    def __init__(self, ion):
        self.ion = ion
        try:
            self.j0 = const.magnetic_ion_j()[self.ion]['j0']
            self.j2 = const.magnetic_ion_j()[self.ion]['j2']
            self.j4 = const.magnetic_ion_j()[self.ion]['j4']
        except ValueError:
            raise ValueError('No such ion was found in database.')

    def calc_mag_form_fac(self, q=None, g=None, qrange=None):
        r'''Calculate the magnetic form factor of an ion.

        Parameters
        ----------
        q : float or list, optional
            An array of values or position at which the form
            factor should be calcuated.

        g : float, optional
            The g-factor, which is 2 is left undefined.

        qrange : float, optional
            The range of q over which the form factor should be
            calculated, if no input array q is provided.

        Returns
        -------
        output : tuple
            (form factor, q, j\ :sub:`0`, j\ :sub:`2`, j\ :sub:`4`)

        Notes
        -----
        The magnetic form factor of an ion is given by:

        .. math:: f(q) = <j_0(q)> + (1-\frac{2}{g})<j_2(q)> \mathrm{(Lovesey,1984)}

        using the 3-gaussian approximation to :math:`f(q)` from the
        International Tables of Crystallography (by J. Brown)


        '''

        if q is None:
            if qrange is None:
                q = np.linspace(0., 2., 2. / 0.025 + 1)
            else:
                q = np.linspace(qrange[0], qrange[1], (qrange[1] - qrange[0]) / 0.025)
        if g is None:
            g = 2.

        x = q / 4. / np.pi

        j0 = (self.j0[0] * np.exp(-self.j0[1] * x ** 2) + self.j0[2] *
              np.exp(-self.j0[3] * x ** 2) + self.j0[4] *
              np.exp(-self.j0[5] * x ** 2) + self.j0[6])

        j2 = x ** 2 * (self.j2[0] * np.exp(-self.j2[1] * x ** 2) +
                       self.j2[2] * np.exp(-self.j2[3] * x ** 2) +
                       self.j2[4] * np.exp(-self.j2[5] * x ** 2) +
                       self.j2[6])

        j4 = x ** 2 * (self.j4[0] * np.exp(-self.j4[1] * x ** 2) +
                       self.j4[2] * np.exp(-self.j4[3] * x ** 2) +
                       self.j4[4] * np.exp(-self.j4[5] * x ** 2) +
                       self.j4[6])

        ff = j0 + (1. - 2. / g) * j2

        return (ff, q, j0, j2, j4)
