r'''Useful constants for neutron scattering calculations, including:

* ``magIonJ()`` : Magnetic Ion j-values
* ``periodicTable()`` : Periodic table values
* ``scatLen()`` : Neutron scattering lengths
* ``joules2meV`` : Joules-to-meV conversion factor
* ``boltzmann_meV_K`` : Boltzmann constant in meV/K

Additional constants are available through scipy.constants.

'''

import os
import json
from scipy import constants


def magIonJ():
    r'''Loads j values for Magnetic ions.

    Parameters
    ----------
    None

    Returns
    -------
    magIonJ : dict
        Database of j-values for magnetic ions

    '''
    with open(os.path.join(os.path.dirname(__file__), "database/mag_ion_j.json"), 'r') as infile:
        return json.load(infile)


def periodicTable():
    r'''Loads periodic table database.
    mass, and long-form name.

    Parameters
    ----------
    None

    Returns
    -------
    periodicTable : dict
        Database of mass, atomic number, density, mass, and name for all elements in the Periodic table

    '''
    with open(os.path.join(os.path.dirname(__file__), "database/periodic_table.json"), 'r') as infile:
        return json.load(infile)


def scatLen():
    r'''Loads neutron scattering lengths.

    Parameters
    ----------
    None

    Returns
    -------
    scatLen : dict
        Database of elements containing the absolute, coherent, incoheret, and
        scattering cross-sections and scattering lengths

    '''
    with open(os.path.join(os.path.dirname(__file__), "database/scat_len.json"), 'r') as infile:
        return json.load(infile)

joules2meV = 1. / constants.physical_constants['electron volt-joule relationship'][0] * 1.e3  # Joules to meV
boltzmann_meV_K = constants.physical_constants['Boltzmann constant in eV/K'][0] * 1.e3  # Boltzmann constant in meV/K
