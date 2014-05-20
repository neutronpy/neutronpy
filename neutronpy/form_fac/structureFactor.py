'''
Created on May 20, 2014

@author: davidfobes
'''
from neutronpy.form_fac.materialClass import * 
import numpy as np


# TODO: Add Debye-Waller Factor support
def str_fac(unitCell, formUnit=1., h=None, k=None, l=None):
    '''
        Calculates the Structural Form Factor of a given unitCell,
        number of formula units (formUnit), and h, k, l

        return: float or numpy array
    '''
    material = Material(data=unitCell, formUnit=formUnit)

    # Determines shape of input variables to build FQ = 0 array
    if type(h) is float and type(k) is float and type(l) is float:
        FQ = 0. * 1j
    else:
        # if one dimension is zero,
        # flatten FQ to 2D
        if type(h) is float:
            FQ = np.zeros(k.shape) * 1j
        elif type(k) is float:
            FQ = np.zeros(l.shape) * 1j
        elif type(l) is float:
            FQ = np.zeros(h.shape) * 1j
        else:
            FQ = np.zeros(h.shape) * 1j

    # construct structure factor
    for atom in material.atoms:
        FQ += atom.b * np.exp(1j * 2. * np.pi *
                              (h * (atom.pos[0] + atom.dpos[0]) +
                               k * (atom.pos[1] + atom.dpos[1]) +
                               l * (atom.pos[2] + atom.dpos[2])))

    return FQ