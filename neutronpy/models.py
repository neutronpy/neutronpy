# -*- coding: utf-8 -*-
r"""Physical Models

"""
import numpy as np


def simple_harmonic_oscillator(p, t):
    r"""Standard equation for a simple harmonic oscillator

    Parameters
    ----------
    p : list
        Parameters, in the following format:

            +-------+----------------------------+
            | p[0]  | Constant background        |
            +-------+----------------------------+
            | p[1]  | Amplitude                  |
            +-------+----------------------------+
            | p[2]  | Period                     |
            +-------+----------------------------+
            | p[3]  | Phase                      |
            +-------+----------------------------+

    t : ndarray
        One dimensional input array

    Returns
    -------
    func : ndarray
        One dimensional array

    """
    return p[0] + p[1] * np.cos(p[2] * t - p[3])


def damped_harmonic_oscillator(p, t):
    r"""Standard equation for a damped harmonic oscillator

    Parameters
    ----------
    p : list
        Parameters, in the following format:

            +-------+----------------------------+
            | p[0]  | Constant background        |
            +-------+----------------------------+
            | p[1]  | Amplitude                  |
            +-------+----------------------------+
            | p[2]  | Period                     |
            +-------+----------------------------+
            | p[3]  | Phase                      |
            +-------+----------------------------+
            | p[4]  | Damping strength           |
            +-------+----------------------------+

    t : ndarray
        One dimensional input array

    Returns
    -------
    func : ndarray
        One dimensional array

    """
    return p[0] + np.exp(-p[4] * t / 2.) * simple_harmonic_oscillator([0, p[1], p[2], p[3]], t)


def acoustic_phonon_dispersion(p, x):
    r"""Standard equation for the dispersion of an acoustic phonon

    Parameters
    ----------
    p : list
        Parameters, in the following format:

            +-------+----------------------------+
            | p[0]  | Dispersion amplitude       |
            +-------+----------------------------+
            | p[1]  | Dispersion period          |
            +-------+----------------------------+


    x : ndarray
        One dimensional input array

    Returns
    -------
    func : ndarray
        One dimensional array

    """
    return np.sqrt(4 * p[0]) * np.abs(np.sin(p[1] * x))


def optical_phonon_disperion():
    r"""Standard equation for the dispersion of an optical phonon

    """
    pass


def ferromagnetic_disperion(p, x):
    r"""Standard equation for the dispersion of a ferromagnetic spin excitation

    Parameters
    ----------
    p : list
        Parameters, in the following format:

            +-------+----------------------------+
            | p[0]  | Dispersion amplitude       |
            +-------+----------------------------+
            | p[1]  | Dispersion period          |
            +-------+----------------------------+

    x : ndarray
        One dimensional input array

    Returns
    -------
    func : ndarray
        One dimensional array

    """
    return 4 * p[0] * (1 - np.cos(p[1] * x))


def antiferromagnetic_disperion(p, x):
    r"""Standard equation for the dispersion of an antiferromagnetic spin excitation

    Parameters
    ----------
    p : list
        Parameters, in the following format:

            +-------+----------------------------+
            | p[0]  | Dispersion amplitude       |
            +-------+----------------------------+
            | p[1]  | Dispersion period          |
            +-------+----------------------------+

    x : ndarray
        One dimensional input array

    Returns
    -------
    func : ndarray
        One dimensional array

    """
    return p[0] * np.abs(np.sin(p[1] * x))
