r'''Calculates common spurions

'''
import numpy as np
from .material import Material
from .energy import Energy


def aluminum(energy=14.7):
    r'''Returns the positions of aluminum rings given a fixed energy

    Parameters
    ----------
    energy : float
        Fixed energy in meV

    Returns
    -------
    rings : str
        Prints a list of the positions in 2theta of the aluminum rings
    '''
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


def currat_axe_peaks():
    pass


def bragg_tails():
    pass
