'''
Created on May 28, 2014

@author: davidfobes
'''

import numpy as np
import math as m
from scipy import special
from scipy.special import erf


def gaussian(p, q):
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
        # Normalization pre-factor
        sigma = p[3 * i + 4] / (2. * m.sqrt(2. * m.log(2.)))
        funct += p[3 * i + 2] / (sigma * m.sqrt(m.pi)) * \
                 np.exp(-(q - p[3 * i + 3]) ** 2 / (2 * sigma ** 2))

    return funct


def lorentzian(p, q):
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


def voigt(p, q):
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


def resolution(p, q, mode='gaussian'):
    '''Purpose: evaluate set of peaks given a resolution [R0, RM] calculated
    with resolution.res_calc() in two dimensions. Gaussian and voigt profiles
    are provided. Parameters are of the format given below.

        p[0]    : flat background term
        p[1]    : sloping background
        p[2]    : area under first gaussian
        p[3]    : x position of first gaussian
        p[4]    : y position of first gaussian
        p[5]    : R0
        p[6]    : RM_xx (Resolution matrix 1st dimension diagonal element)
        p[7]    : RM_yy (Resolution matrix 2nd dimension diagonal element)
        p[8]    : RM_xy (Resolution matrix off diagonal element)
        etc.
    '''
    x, y = q

    funct = p[0] + p[1] * (x + y)

    if mode == 'gaussian':
        for i in range(len(p[2:] / 7)):
            # Normalization pre-factor
            N = (np.sqrt(p[7 * i + 6]) * np.sqrt(p[7 * i + 7] - p[7 * i + 8] ** 2 / p[7 * i + 6])) / (2. * np.pi)
            funct += p[7 * i + 2] * p[7 * i + 5] * N * np.exp(-1. / 2. * (p[7 * i + 6] * (x - p[7 * i + 3]) ** 2 +
                                                                          p[7 * i + 7] * (y - p[7 * i + 4]) ** 2 +
                                                                          2. * p[7 * i + 8] * (x - p[7 * i + 3]) * (y - p[7 * i + 4])))

    return funct


def gaussian_ring(p, q):
    '''Purpose: evaluate terms and derivatives of function
    for non-linear least-squares search with form
    of normalized Gaussian ellipse(s) in 2 dimensions

        p[0]    : flat background term
        p[1]    : sloping background
        p[2]    : area under first gaussian ellipse
        p[3]    : x position of first gaussian ellipse
        p[4]    : y position of first gaussian ellipse
        p[5]    : radius of the first gaussian ellipse
        p[6]    : eccentricity of the first gaussian ellipse
        p[7]    : full width half maximum of first gaussian ellipse
        etc.
    '''
    x, y = q

    funct = p[0] + p[1] * (x + y)

    for i in range(len(p[2:] / 6)):
        # Normalization pre-factor
        N = 1. / ((2. * np.pi / p[6 * i + 6] ** 2) * (p[6 * i + 7] ** 2 / (8. * np.log(2.))) *
                  np.exp(-4. * np.log(2.) * p[6 * i + 5] ** 2 / p[6 * i + 7] ** 2) +
                  np.sqrt(np.pi / np.log(2.)) * p[6 * i + 5] *
                  (1. + erf(4. * np.sqrt(np.log(2.)) * p[6 * i + 5] / p[6 * i + 7])))

        funct += p[6 * i + 2] * N * np.exp(-4. * np.log(2.) * (np.sqrt((x - p[6 * i + 3]) ** 2 +
                                                                       p[6 * i + 6] ** 2 * (y - p[6 * i + 4]) ** 2) -
                                                               p[6 * i + 5]) ** 2 / p[6 * i + 7])

    return funct
