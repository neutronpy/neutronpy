r'''Useful constants for neutron scattering calculations, including:

* ``magnetic_ion_j()`` : Magnetic Ion j-values
* ``periodic_table()`` : Periodic table values
* ``scattering_lengths()`` : Neutron scattering lengths
* ``JOULES_TO_MEV`` : Joules-to-meV conversion factor
* ``BOLTZMANN_IN_MEV_K`` : Boltzmann constant in meV/K

Additional constants are available through scipy.constants.

'''

import os
import json
from scipy import constants as consts


def magnetic_ion_j():
    r'''Loads j values for Magnetic ions.

    Parameters
    ----------
    None

    Returns
    -------
    magnetic_ion_j : dict
        Database of j-values for magnetic ions

    '''
    with open(os.path.join(os.path.dirname(__file__),
                           "database/mag_ion_j.json"), 'r') as infile:
        return json.load(infile)


def periodic_table():
    r'''Loads periodic table database.
    mass, and long-form name.

    Parameters
    ----------
    None

    Returns
    -------
    periodic_table : dict
        Database of mass, atomic number, density, mass, and name for all
        elements in the Periodic table

    '''
    with open(os.path.join(os.path.dirname(__file__),
                           "database/periodic_table.json"), 'r') as infile:
        return json.load(infile)


def scattering_lengths():
    r'''Loads neutron scattering lengths.

    Parameters
    ----------
    None

    Returns
    -------
    scattering_lengths : dict
        Database of elements containing the absolute, coherent, incoheret, and
        scattering cross-sections and scattering lengths

    '''
    with open(os.path.join(os.path.dirname(__file__),
                           "database/scat_len.json"), 'r') as infile:
        return json.load(infile)

JOULES_TO_MEV = 1. / consts.physical_constants\
    ['electron volt-joule relationship'][0] * 1.e3  # Joules to meV
BOLTZMANN_IN_MEV_K = consts.physical_constants['Boltzmann constant in eV/K'][0]\
                        * 1.e3  # Boltzmann constant in meV/K
N_A = consts.N_A
