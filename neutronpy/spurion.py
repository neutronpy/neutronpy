# -*- coding: utf-8 -*-
r"""Calculates common spurions

"""
import warnings

import numpy as np

from .crystal import Material
from .energy import Energy


def aluminum(energy=14.7):
    r"""Returns the positions of aluminum rings given a fixed energy

    Parameters
    ----------
    energy : float
        Fixed energy in meV

    Returns
    -------
    rings : str
        Prints a list of the positions in 2theta of the aluminum rings
    """
    e = Energy(energy=energy)
    struct = {'name': 'Al',
              'composition': [dict(ion='Al', pos=[0, 0, 0])],
              'debye-waller': False,
              'massNorm': False,
              'lattice': dict(abc=[4.0495, 4.0495, 4.0495], abg=[90, 90, 90]),
              'formulaUnits': 1.,
              'wavelength': e.wavelength,
              'space_group': 'Fm-3m'}

    reflections = ([1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1])

    struct_obj = Material(struct)
    wavelengths = [e.wavelength / 3, e.wavelength / 2, e.wavelength]
    print('(h, k, l)  2theta  |F|^2  wavelength')
    print('------------------------------------')

    hkl = []
    two_theta = []
    wavelength_fraction = []
    str_fac = []
    for wavelength in wavelengths:
        for pos in reflections:
            wavelength_fraction.append('lambda/{0:.0f}'.format(np.round(e.wavelength / wavelength)))
            hkl.append(str(pos))
            two_theta.append(struct_obj.get_two_theta(pos, wavelength))
            str_fac.append(np.abs(struct_obj.calc_nuc_str_fac(pos)) ** 2)
    hkl = np.array(hkl)
    two_theta = np.array(two_theta)
    wavelength_fraction = np.array(wavelength_fraction)
    str_fac = np.array(str_fac)

    ind = two_theta.argsort()
    for pos, tt, i0, lam in zip(hkl[ind], two_theta[ind], str_fac[ind], wavelength_fraction[ind]):
        print(pos, '{0:.4f}'.format(tt), '{0:.0f}'.format(i0), lam)


def currat_axe_peaks(instrument, scan, bragg_positions, angle_tol=1):
    r"""Notifies if desired scan may contain Currat Axe spurious scattering
    given the instrument and position of nearby Bragg peaks

    Parameters
    ----------
    instrument : object
        Instrument on which scan will be performed

    scan : list
        Desired Scan given by [[qx0,qy0,e0],[qx1,qy1,e1],npts],
        where qx and qy are along u and v, respectively, in r.l.u. and npts is
        the number of points in the scan.

        For example, if u=[1,1,0], and v=[0,0,1] and you want to scan from
        [0.8, 0.8, 0] to [1.2, 1.2, 0] at 3 meV, with a 0.025 step then
        your scan would be given by [[0.8, 0, 3],[1.2,0,3],17].

    bragg_positions : list
        List of Bragg positions nearby to the scan you wish to perform,
        in the format [[qx, qy],[qx,qy],...]

    angle_tol : float
        How close in angle the motors should be to satistfy the scattering
        condition

    """

    hkl0 = list(instrument.sample.u * scan[0][0] + instrument.sample.v * scan[0][1])
    hkle0 = np.insert(hkl0, 3, scan[0][2])

    hkl1 = list(instrument.sample.u * scan[1][0] + instrument.sample.v * scan[1][1])
    hkle1 = np.insert(hkl1, 3, scan[1][2])

    vec = (hkle1 - hkle0)
    dvecs = np.linspace(0, 1, scan[2])
    hkle_scan = [hkle0 + vec * dvec for dvec in dvecs]

    angles = []
    for hkle in hkle_scan:
        angles.append(instrument.get_angles_and_Q(hkle)[0][2:4])

    bragg_angles = []
    for peak in bragg_positions:
        _peak = np.insert(peak, 3, 0)
        bragg_angles.append(instrument.get_angles_and_Q(_peak)[0][2:4])

    for n, angle in enumerate(angles):
        for bragg_angle in bragg_angles:
            if np.all(np.abs(bragg_angle - angle) <= angle_tol):
                warnings.warn('WARNING: YOUR SCAN MAY CONTAIN CURRAT-AXE SCATTERING AT {0}'.format(hkle_scan[n]))


def bragg_tails():
    pass
