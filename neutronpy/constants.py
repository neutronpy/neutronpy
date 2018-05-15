# -*- coding: utf-8 -*-
r"""Useful constants for neutron scattering calculations, including:

* ``magnetic_form_factors()`` : Magnetic Ion j-values
* ``periodic_table()`` : Periodic table values
* ``scattering_lengths()`` : Neutron scattering lengths
* ``symmetry()`` : Space group information
* ``JOULES_TO_MEV`` : Joules-to-meV conversion factor
* ``BOLTZMANN_IN_MEV_K`` : Boltzmann constant in meV/K
* ``N_A`` : Avogadro constant
* ``neutron_mass`` : Mass of a neutron in grams
* ``e`` : Electric charge of an electron in Coulombs

"""
import json
import os


def magnetic_ion_j():
    r"""Loads j values for Magnetic ions.

    Parameters
    ----------
    None

    Returns
    -------
    magnetic_ion_j : dict
        Database of j-values for magnetic ions

    """
    with open(os.path.join(os.path.dirname(__file__),
                           "database/magnetic_form_factors.json"), 'r') as infile:
        return json.load(infile)


def periodic_table():
    r"""Loads periodic table database.
    mass, and long-form name.

    Parameters
    ----------
    None

    Returns
    -------
    periodic_table : dict
        Database of mass, atomic number, density, mass, and name for all
        elements in the Periodic table

    """
    with open(os.path.join(os.path.dirname(__file__),
                           "database/periodic_table.json"), 'r') as infile:
        return json.load(infile)


def scattering_lengths():
    r"""Loads neutron scattering lengths.

    Parameters
    ----------
    None

    Returns
    -------
    scattering_lengths : dict
        Database of elements containing the absolute, coherent, incoheret, and
        scattering cross-sections and scattering lengths

    """
    with open(os.path.join(os.path.dirname(__file__),
                           "database/scattering_lengths.json"), 'r') as infile:
        return json.load(infile)


def symmetry():
    r"""Loads crystal lattice space groups.

    Parameters
    ----------
    None

    Returns
    -------
    lattice_space_groups : dict
        Database of 230 crystal lattice space groups and their generators

    """
    with open(os.path.join(os.path.dirname(__file__),
                           "database/symmetry.json"), 'r') as infile:
        return json.load(infile)


JOULES_TO_MEV = 1. / 1.6021766208e-19 * 1.e3  # Joules to meV
BOLTZMANN_IN_MEV_K = 8.6173303e-05 * 1.e3  # Boltzmann constant in meV/K
N_A = 6.022140857e+23
neutron_mass = 1.674927211e-24  # mass of a neutron in grams
hbar = 1.054571628e-34  # hbar in m2 kg / s
e = 1.602176487e-19  # coulombs
