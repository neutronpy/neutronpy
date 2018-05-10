# -*- coding: utf-8 -*-
r"""Structure Factors

NuclearStructureFactor
MagneticStructureFactor
MagneticFormFactor

"""
import numpy as np

from ..constants import magnetic_ion_j


class NuclearStructureFactor(object):
    r"""Class containing nuclear structure factor calculator

    Methods
    -------
    calc_nuc_str_fac

    """
    def calc_nuc_str_fac(self, hkl):
        r"""Calculates the structural form factor of the material.

        Parameters
        ----------
        hkl : tuple of floats, or tuple of array-like
            Reciprocal lattice positions at which the structure
            factor should be calculated

        Returns
        -------
         NSF : float or ndarray
             Nuclear structure factor at the position or positions specified

        Notes
        -----

        """

        h, k, l = hkl

        # Ensures input arrays are complex ndarrays
        if isinstance(h, (np.ndarray, list, tuple)):
            h = np.array(h).astype(complex)
        if isinstance(k, (np.ndarray, list, tuple)):
            k = np.array(k).astype(complex)
        if isinstance(l, (np.ndarray, list, tuple)):
            l = np.array(l).astype(complex)

        # construct structure factor
        NSF = 0 * 1j
        for atom in self.atoms:
            NSF += atom.occupancy * atom.b * np.exp(1j * 2. * np.pi * (h * atom.pos[0] + k * atom.pos[1] + l * atom.pos[2])) * \
                np.exp(-8 * np.pi ** 2 * atom.Uiso * np.sin(np.deg2rad(self.get_two_theta(atom.pos, self.wavelength) / 2.)) ** 2 / self.wavelength ** 2) * \
                np.exp(-np.float(np.dot(np.dot(atom.pos, atom.Uaniso), atom.pos)))

        return NSF


class MagneticFormFactor(object):
    r"""Class defining a magnetic ion.

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
    """

    def __init__(self, ion):
        self.ion = ion
        try:
            self.j0 = magnetic_ion_j()[self.ion]['j0']
            self.j2 = magnetic_ion_j()[self.ion]['j2']
            self.j4 = magnetic_ion_j()[self.ion]['j4']
        except ValueError:
            raise ValueError('No such ion was found in database.')

    def __repr__(self):
        return "MagneticFormFactor('{0}')".format(self.ion)

    def calc_mag_form_fac(self, q=None, g=None, qrange=None):
        r"""Calculate the magnetic form factor of an ion.

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

        .. math:: f(q) = <j_0(q)> + (\frac{2}{g}-1)<j_2(q)> \mathrm{(Jensen and Mackintosh,1991)}

        using the 3-gaussian approximation to :math:`f(q)` from the
        International Tables of Crystallography (by J. Brown)


        """

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

        ff = j0 + (2. / g - 1.) * j2

        return ff, q, j0, j2, j4


class MagneticStructureFactor(object):
    r"""Class containing magnetic structure factor calculator

    Methods
    -------
    calc_mag_int_vec
    calc_mag_str_fac

    """
    def calc_mag_int_vec(self):
        r"""Calculates magnetic interaction vector
        """
        pass

    def calc_mag_str_fac(self):
        r"""Calculates magnetic structure factor
        """
        pass
