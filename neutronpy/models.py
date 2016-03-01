r'''Physical Models

'''
import numpy as np


def simple_harmonic_oscillator(p, t):
    r'''Standard equation for a simple harmonic oscillator
    '''
    return p[0] + p[1] * np.cos(p[2] * t - p[3])


def damped_harmonic_oscillator(p,t):
    r'''Standard equation for a damped harmonic oscillator
    '''
    return np.exp(-p[4] * t / 2) * simple_harmonic_oscillator(p, t)


def acoustic_phonon_dispersion(p, x):
    r'''Standard equation for the dispersion of an acoustic phonon
    '''
    return np.sqrt(4 * p[0]) * np.abs(np.sin(p[1] * x))


def optical_phonon_disperions():
    r'''Standard equation for the dispersion of an optical phonon
    '''
    pass


def ferromagnetic_disperions(p, x):
    r'''Standard equation for the dispersion of a ferromagnetic spin excitation
    '''
    return 4 * p[0] * (1 - np.cos(p[1] * x))


def antiferromagnetic_disperions(p, x):
    r'''Standard equation for the dispersion of an antiferromagnetic spin excitation
    '''
    return p[0] * np.abs(np.sin(p[1] * x))
