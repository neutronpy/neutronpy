'''
Created on May 28, 2014

@author: davidfobes
'''

import numpy as np
import math as m
from scipy import special


def gaussian(q, p):
    '''Purpose: evaluate terms and derivatives of function
    for non-linear least-squares search with form of Gaussian

        p[0]: constant background
        p[1]: background slope
        p[2]: integrated intensity of peak 1
        p[3]: position of peak 1
        p[4]: fwhm of peak 1
        p[5]: integrated intensity of peak 2
        p[6]: position of peak 2
        p[7]: fwhm of peak 2
        etc.
    '''
    funct = p[0] + p[1] * q

    for i in range(len(p[2:]) / 3):
        sigma = p[3 * i + 4] / (2. * m.sqrt(2. * m.log(2.)))
        funct += p[3 * i + 2] / (sigma * m.sqrt(m.pi)) * \
                 np.exp(-(q - p[3 * i + 3]) ** 2 / (2 * sigma ** 2))

    return funct


def lorentzian(q, p):
    '''Purpose: evaluate terms and derivatives of function for
    non-linear least-squares search with form of Lorentzian

        p[0]: constant background
        p[1]: background slope
        p[2]: integrated intensity of peak 1
        p[3]: position of peak 1
        p[4]: fwhm of peak 1
        p[5]: integrated intensity of peak 2
        p[6]: position of peak 2
        p[7]: fwhm of peak 2
        etc.
    '''

    funct = p[0] + p[1] * q

    for i in range(len(p[2:]) / 3):
        funct += p[3 * i + 2] / m.pi * 0.5 * p[3 * i + 4] / \
                 ((q - p[3 * i + 3]) ** 2 * (0.5 * p[3 * i + 4]) ** 2)

    return funct


def voigt(q, p):
    '''Purpose: evaluate set of peaks in the form of Voigt functions
    describing Lorentzian function convoluted with the Gaussian resolution.
    Requires Gaussian resolution width in addition to three Lorentzian
    parameters for each peak.

        p[0]    : flat background term
        p[1]    : sloping background
        p[2]    : area under first Lorentzian
        p[3]    : position of first Lorentzian
        p[4]    : FWHM of first Lorentzian
        p[5]    : FWHM of first Gaussian resolution
        p[6]    : area under second Lorentzian
        etc.
    '''

    funct = p[0] + p[1] * q

    for i in range(len(p[2:] / 4)):
        funct += p[4 * i + 2] / special.wofz(np.zeros((len(q))) + 1j *
                                             np.sqrt(np.log(2.)) *
                                             p[4 * i + 5]).real * \
                                special.wofz(2 * np.sqrt(np.log(2.)) *
                                             (q - p[4 * i + 3]) /
                                             p[4 * i + 4] +
                                             1j * np.sqrt(np.log(2.)) *
                                             p[4 * i + 5]).real

        return funct


def gaussianResolution(q, p):
    pass


def gaussianRing(q, p):
    '''Purpose: evaluate terms and derivatives of function
    for non-linear least-squares search with form
    of normalized Gaussian ring(s) in 2 dimensions

    '''
