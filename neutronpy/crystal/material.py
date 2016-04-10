# -*- coding: utf-8 -*-
r'''Material constructor

'''
import numpy as np
import neutronpy.constants as const
from .atom import Atom, MagneticAtom
from .plot import PlotMaterial
from .sample import Sample
from .structure_factors import NuclearStructureFactor, MagneticStructureFactor
from .symmetry import SpaceGroup
#from ..scattering.pattern import HKLGenerator


class MagneticUnitCell(Sample):
    r'''Class defining a magnetic unit cell

    '''
    def __init__(self, unit_cell):
        if 'chirality' not in unit_cell:
            self.chirality = 0
        if 'ph' not in unit_cell:
            self.phase = 0
        self.atoms = []
        if 'space_group' in unit_cell:
            self.space_group = SpaceGroup(unit_cell['space_group'])
            for atom in unit_cell['atoms']:
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(MagneticAtom(atom['ion'],
                                                   pos,
                                                   atom['moment'],
                                                   atom['occupancy']))
        else:
            for atom in unit_cell['atoms']:
                self.atoms.append(MagneticAtom(atom['ion'],
                                               atom['pos'],
                                               atom['moment'],
                                               atom['occupancy']))

        self.propagation_vector = unit_cell['propagation_vector']

        a, b, c = unit_cell['lattice']['abc']
        alpha, beta, gamma = unit_cell['lattice']['abg']
        super(MagneticUnitCell, self).__init__(a, b, c, alpha, beta, gamma)


class Material(Sample, NuclearStructureFactor, MagneticStructureFactor, PlotMaterial):
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

    'composition' : list of dicts
        For each atom in the unit cell, you must provide:

            'ion' : str
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

    'formulaUnits' : float
        Number of formula units to use in the calculation

    'lattice' : dict

        'abc' : lattice constants of unit cell

        'abg' : lattice angles of unit cell

    'space_group' : str or int
        Hermann–Mauguin symbol or international space group number

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
    u
    v

    Methods
    -------
    calc_nuc_str_fac
    calc_mag_str_fac
    calc_mag_int_vec
    calc_incoh_elas_xs
    calc_optimal_thickness
    plot_unit_cell
    get_angle_between_planes
    get_d_spacing
    get_q
    get_two_theta
    N_atoms
    apply_scattering_rules
    find_equivalent_positions
    find_site_multiplicity
    generate_hkl_positions

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
            a, b, c = crystal['lattice']['abc']
            alpha, beta, gamma = crystal['lattice']['abg']

        if 'wavelength' in crystal:
            self.wavelength = crystal['wavelength']
        else:
            self.wavelength = 2.359

        if 'space_group' in crystal:
            self.space_group = SpaceGroup(crystal['space_group'])
            self.atoms = []
            for atom in crystal['composition']:
                if 'Uiso' not in atom:
                    atom['Uiso'] = 0
                if 'Uaniso' not in item:
                    atom['Uaniso'] = np.matrix(np.zeros((3, 3)))
                if 'occupancy' not in item:
                    atom['occupancy'] = 1.
                symmetrized_positions = self.space_group.symmetrize_position(atom['pos'])
                for pos in symmetrized_positions:
                    self.atoms.append(Atom(atom['ion'],
                                           pos,
                                           atom['occupancy'],
                                           self.Mcell,
                                           crystal['massNorm'],
                                           atom['Uiso'],
                                           atom['Uaniso']))
        else:
            self.atoms = []
            for item in crystal['composition']:
                if 'Uiso' not in item:
                    item['Uiso'] = 0
                if 'Uaniso' not in item:
                    item['Uaniso'] = np.matrix(np.zeros((3, 3)))
                if 'occupancy' not in item:
                    item['occupancy'] = 1.
                self.atoms.append(Atom(item['ion'],
                                       item['pos'],
                                       item['occupancy'],
                                       self.Mcell,
                                       crystal['massNorm'],
                                       item['Uiso'],
                                       item['Uaniso']))

        if 'magnetic_unit_cell' in crystal:
            self.magnetic_unit_cell = MagneticUnitCell(crystal['magnetic_cell'])

        if 'mosaic' not in crystal:
            crystal['mosaic'] = None
        if 'vmosaic' not in crystal:
            crystal['vmosaic'] = None
        if 'u' not in crystal:
            crystal['u'] = None
        if 'v' not in crystal:
            crystal['v'] = None
        if 'dir' not in crystal:
            crystal['dir'] = 1

        super(Material, self).__init__(a, b, c, alpha, beta, gamma, crystal['mosaic'], crystal['vmosaic'], crystal['dir'], crystal['u'], crystal['v'])

    @property
    def total_scattering_cross_section(self):
        r'''Returns total scattering cross-section of unit cell
        '''
        total = 0
        for atom in self.atoms:
            total += (atom.coh_xs + atom.inc_xs)
        return total

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

    def calc_incoh_elas_xs(self, mass=None):
        r'''Calculates the incoherent elastic cross section.

        Parameters
        ----------
        mass : float (optional)
            The mass of the sample in grams (Default: None)

        Returns
        -------
        INC_XS : float
            Incoherent Elastic Cross Section

        '''

        INC_XS = 0
        for atom in self.atoms:
            INC_XS += atom.inc_xs * np.exp(-8 * np.pi ** 2 * atom.Uiso * np.sin(np.deg2rad(self.get_two_theta(atom.pos, self.wavelength) / 2.)) ** 2 / self.wavelength ** 2)

        if mass is not None:
            return self.N_atoms(mass) / (4 * np.pi) * INC_XS
        else:
            return INC_XS / (4 * np.pi)



