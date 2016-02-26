r'''Calculates lattice and magnetic structure/form factors

'''
from numbers import Number
import numpy as np
import neutronpy.constants as const
from .lattice import Lattice


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
        Occupancy of the _Atom (*e.g.* if there is partial occupancy from
        doping)
    Mcell : float, optional
        The mass of the unit cell. If assigned, normalize scattering lengths to
        the square-root of the mass of the atom

    Returns
    -------
    output : object
        Atom object defining an individual atom in a unit cell of a single
        crystal

    '''
    def __init__(self, ion, pos, occupancy=1., Mcell=None, massNorm=False, Uiso=0, Uaniso=np.zeros((3, 3))):
        self.ion = ion
        self.pos = np.array(pos)
        self.occupancy = occupancy
        self.Mcell = Mcell
        self.Uiso = Uiso
        self.Uaniso = np.matrix(Uaniso)

        if massNorm is True:
            self.mass = const.periodic_table()[ion]['mass']
            self.b = (const.scattering_lengths()[ion]['Coh b'] * self.occupancy * self.Mcell / np.sqrt(self.mass))
        else:
            self.b = const.scattering_lengths()[ion]['Coh b'] / 10.

        self.coh_xs = const.scattering_lengths()[ion]['Coh xs']
        self.inc_xs = const.scattering_lengths()[ion]['Inc xs']
        self.abs_xs = const.scattering_lengths()[ion]['Abs xs']


class Material(Lattice):
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
                              'Uiso': float,
                              'Uaniso': matrix(3,3)
                              'occupancy': float}],
             'massNorm': bool,
             'formulaUnits': float,
             'lattice': {'abc': [float, float, float],
                         'abg': [float, float, float]}

    The following are valid options for the data dictionary:

    'name' : str
        Name of the structure

    'composition': list of dicts
        For each atom in the unit cell, you must provide:

            'ion': str
                Name of the atom. Needed for mass and scattering length

            'pos' : list of 3 floats
                x, y, z position of the atom within the unit cell in normalized
                units

            'dpos' : list of 3 floats
                x, y, z displacements of the atom, for Debye-Waller factor, in
                normalized units

            'occupancy' : float
                Occupancy of the site, e.g. if atoms only partially occupy this
                site

    'Uiso' : bool
        Include Debye-Waller in calculation with isotropic U

    'Uaniso' : bool
        Include Debye-Waller in calculation with anisotropic U

    'massNorm' : bool
        Normalize calculations to mass of atoms

    'formulaUnits': float
        Number of formula units to use in the calculation

    'lattice' : dict
        'abc': lattice constants of unit cell
        'abg': lattice angles of unit cell

    Returns
    -------
    output : object
        Material Object defining a single crystal.

    Attributes
    ----------
    volume
    total_scattering_cross_section
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
    calc_str_fac
    calc_optimal_thickness
    plot_unit_cell
    get_angle_between_planes
    get_d_spacing
    get_q
    get_two_theta
    N_atoms

    '''

    def __init__(self, crystal):
        if 'formulaUnits' not in crystal:
            crystal['formulaUnits'] = 1.

        self.muCell = 0.
        for item in crystal['composition']:
            if 'occupancy' not in item:
                item['occupancy'] = 1.
            self.muCell += const.periodic_table()[item['ion']]['mass'] * item['occupancy']

        self.Mcell = self.muCell * crystal['formulaUnits']

        if 'lattice' in crystal:
            self.abc = crystal['lattice']['abc']
            self.abg = crystal['lattice']['abg']

        if 'wavelength' in crystal:
            self.wavelength = crystal['wavelength']
        else:
            self.wavelength = 2.359

        self.atoms = []
        for item in crystal['composition']:
            if 'Uiso' not in item:
                item['Uiso'] = 0
            if 'Uaniso' not in item:
                item['Uaniso'] = np.matrix(np.zeros((3, 3)))
            if 'occupancy' not in item:
                item['occupancy'] = 1.
            self.atoms.append(_Atom(item['ion'],
                                    item['pos'],
                                    item['occupancy'],
                                    self.Mcell,
                                    crystal['massNorm'],
                                    item['Uiso'],
                                    item['Uaniso']))
        super(Material, self).__init__(self.abc, self.abg)

    @property
    def total_scattering_cross_section(self):
        r'''Returns total scattering cross-section of unit cell
        '''
        total = 0
        for atom in self.atoms:
            total += (atom.coh_xs + atom.inc_xs)
        return total

    def calc_str_fac(self, hkl):
        r'''Calculates the structural form factor of the material.

        Parameters
        ----------
        hkl : tuple of floats, or tuple of array-like
            Reciprocal lattice positions at which the structure
            factor should be calculated

        Returns
        -------
         FQ : float or ndarray
             Structure factor at the position or positions specified

        '''

        h, k, l = hkl

        # Ensures input arrays are complex ndarrays
        if isinstance(h, (np.ndarray, list, tuple)):
            h = np.array(h).astype(complex)
        if isinstance(k, (np.ndarray, list, tuple)):
            k = np.array(k).astype(complex)
        if isinstance(l, (np.ndarray, list, tuple)):
            l = np.array(l).astype(complex)

        # Determines shape of input variables to build FQ = 0 array
        _dims = h + k + l
        if isinstance(_dims, Number):
            FQ = 0 * 1j
        else:
            FQ = np.zeros(_dims.shape, dtype=np.complex)

        # construct structure factor
        for atom in self.atoms:
            FQ += atom.occupancy * atom.b * np.exp(1j * 2. * np.pi * (h * atom.pos[0] + k * atom.pos[1] + l * atom.pos[2])) * \
                np.exp(-8 * np.pi ** 2 * atom.Uiso * np.sin(np.deg2rad(self.get_two_theta(atom.pos, self.wavelength) / 2.)) ** 2 / self.wavelength ** 2) * \
                np.exp(-np.float(np.dot(np.dot(atom.pos, atom.Uaniso), atom.pos)))

        return FQ

    def plot_unit_cell(self):
        r'''Plots the unit cell and atoms of the material.

        '''

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport
        from itertools import product, combinations

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # draw unit cell
        for s, e in combinations(np.array(list(product([0, self.abc[0]], [0, self.abc[1]], [0, self.abc[2]]))), 2):
            if np.sum(np.abs(s - e)) in self.abc:
                ax.plot3D(*zip(s, e), color="b")

        # plot atoms
        x, y, z, m = [], [], [], []
        for item in self.atoms:
            x.append(item.pos[0] * self.abc[0])
            y.append(item.pos[1] * self.abc[1])
            z.append(item.pos[2] * self.abc[2])
            m.append(item.mass)

        ax.scatter(x, y, z, s=m)

        plt.axis('scaled')
        plt.axis('off')

        plt.show()

    def N_atoms(self, mass):
        r'''Number of atoms in the defined Material, given the mass of the sample.

        Parameters
        ----------
        mass : float
            The mass of the sample in grams.

        Returns
        -------
        Natoms : int
            The number of atoms of the material based on the mass of the sample.

        '''
        return const.N_A * mass / self.muCell

    def calc_optimal_thickness(self, energy=25.3, transmission=1 / np.exp(1)):
        r'''Calculates the optimal sample thickess to avoid problems with
        extinction, multiple coherent scattering and absorption.

        Parameters
        ----------
        energy : float, optional
            The energy of the incident beam in meV. Default: 25.3 meV.

        transmission: float, optional
            The transmission through the material in decimal percentage,
            :math:`0 < T < 1.0`. Default: :math:`1/e`.

        Returns
        -------
        thickness : float
            Returns the optimal thickness of the sample in cm

        Notes
        -----
        The total transmission of neutrons through a material is defined as

        .. math:: T = \frac{I}{I_0} = e^{-\Sigma_T d},

        where :math:`\Sigma_T` is the total scattering cross-section, *i.e.*

        .. math:: \Sigma_T = \Sigma_{coh} + \Sigma_{inc} + \Sigma_{abs},

        and :math:`d` is the thickness in cm.

        Scattered intensity is thus defined to be

        .. math:: I_s \propto dT\left(\frac{d\Sigma_{coh}}{d\Omega}\right) \propto d e^{-\Sigma_T d}.

        :math:`I_s` is therefore a maximum when :math:`d=1/\Sigma_T`, resulting
        a transmission of approximately :math:`T=37\%`. This is valid when the
        coherent cross-section is much less than the total cross-section,
        *i.e.*

        .. math:: \Sigma_{coh} \ll \Sigma_T \approx \Sigma_{inc} + \Sigma{abs}.

        However, if the coherent cross-section of the material is the dominant
        part of the total scattering cross-section, *i.e.*
        :math:`\Sigma_T \approx \Sigma_{coh}`, then :math:`d = 1/\Sigma_T` is
        too large and there will be a problem with multiple scattering.
        Therefore a transmission of :math:`T\geq90\%` is desirable.

        '''

        sigma_coh = np.sum([atom.occupancy * atom.coh_xs for atom in self.atoms])
        sigma_inc = np.sum([atom.occupancy * atom.inc_xs for atom in self.atoms])
        sigma_abs = np.sum([atom.occupancy * atom.abs_xs for atom in self.atoms])

        sigma_T = (sigma_coh + sigma_inc + sigma_abs * np.sqrt(25.3 / energy)) / self.volume

        return -np.log(transmission) / sigma_T


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
                q = np.linspace(qrange[0], qrange[1], (qrange[1] - qrange[0]) / 0.025 + 1)
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
