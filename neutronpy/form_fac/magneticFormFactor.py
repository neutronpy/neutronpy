'''
Created on May 20, 2014

@author: davidfobes
'''
import neutronpy.constants
import numpy as np


def mag_fac(ion, q=None, g=None, qrange=None):
    '''
        Calculate the magnetic form factor of ion given by
        
        f(q) = <j0(q)> + (1-2/g)<j2(q)> (Lovesey,1984)
        
        using the 3-gaussian approximation to fl(q) from the
        International Tables of Crystallography (by J. Brown)
        
        Returns the tuple:
            (form factor, q, j0, j2, j4)
    '''
    mat = Ion(ion)
    
    if q is None:
        if qrange is None:
            q = np.linspace(0., 2., 2. / 0.025 + 1)
        else:
            q = np.linspace(qrange[0], qrange[1], (qrange[1] - qrange[0]) / 0.025)
    if g is None:
        g = 2.
    
    x = q / 4. / np.pi
    
    j0 = (mat.j0[0] * np.exp(-mat.j0[1] * x ** 2) + mat.j0[2] * np.exp(-mat.j0[3] * x ** 2) + mat.j0[4] * np.exp(-mat.j0[5] * x ** 2) + mat.j0[6])
    j2 = x ** 2 * (mat.j2[0] * np.exp(-mat.j2[1] * x ** 2) + mat.j2[2] * np.exp(-mat.j2[3] * x ** 2) + mat.j2[4] * np.exp(-mat.j2[5] * x ** 2) + mat.j2[6])
    j4 = x ** 2 * (mat.j4[0] * np.exp(-mat.j4[1] * x ** 2) + mat.j4[2] * np.exp(-mat.j4[3] * x ** 2) + mat.j4[4] * np.exp(-mat.j4[5] * x ** 2) + mat.j4[6])

    ff = j0 + (1. - 2. / g) * j2

    return (ff, q, j0, j2, j4)
