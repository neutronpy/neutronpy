# -*- coding: utf-8 -*-
r""" Class to calculate the energy of a neutron in various common units
"""
import numpy as np
from scipy.constants import h, hbar, k, m_n

from .constants import JOULES_TO_MEV


class Energy(object):
    u"""Class containing the most commonly used properties of a neutron beam
    given some initial input, e.g. energy, wavelength, velocity, wavevector,
    temperature, or frequency. At least one input must be supplied.

    Parameters
    ----------
    energy : float
        Neutron energy in millielectron volts (meV)
    wavelength : float
        Neutron wavelength in angstroms (Å)
    velocity : float
        Neutron velocity in meters per second (m/s)
    wavevector : float
        Neutron wavevector k in inverse angstroms (1/Å)
    temperature : float
        Neutron temperature in kelvin (K)
    frequency : float
        Neutron frequency in terahertz (THz)

    Returns
    -------
    Energy object
        The energy object containing the properties of the neutron beam

    Attributes
    ----------
    energy
    wavelength
    wavevector
    velocity
    temperature
    frequency
    values
    """

    def __init__(self, energy=None, wavelength=None, velocity=None, wavevector=None, temperature=None, frequency=None):
        self._update_values(energy, wavelength, velocity, wavevector, temperature, frequency)

    def __str__(self):
        return self.values

    def __repr__(self):
        return "Energy({0})".format(self.energy)

    def __eq__(self, right):
        return abs(self.energy - right.energy) < 1e-6

    def __ne__(self, right):
        return not self.__eq__(right)

    def _update_values(self, energy=None, wavelength=None, velocity=None, wavevector=None, temperature=None,
                       frequency=None):
        try:
            if energy is None:
                if wavelength is not None:
                    self.en = h ** 2. / (2. * m_n * (wavelength / 1.0e10) ** 2.) * JOULES_TO_MEV
                elif velocity is not None:
                    self.en = 1. / 2. * m_n * velocity ** 2 * JOULES_TO_MEV
                elif wavevector is not None:
                    self.en = (h ** 2 / (2. * m_n * ((2. * np.pi / wavevector) / 1.e10) ** 2) * JOULES_TO_MEV)
                elif temperature is not None:
                    self.en = k * temperature * JOULES_TO_MEV
                elif frequency is not None:
                    self.en = (hbar * frequency * 2. * np.pi * JOULES_TO_MEV * 1.e12)
            else:
                self.en = energy

            if np.any(self.energy == 0.0):
                if isinstance(self.energy, np.ndarray):
                    self.wavelen = np.full(self.energy.shape, np.nan)
                    self.wavevec = np.zeros(self.energy.shape)

                    self.wavelen[self.energy != 0.0] = np.sqrt(h ** 2 / (2. * m_n * self.energy[self.energy !=0] / JOULES_TO_MEV)) * 1.e10
                    self.wavevec[self.energy != 0.0] = 2. * np.pi / self.wavevec[self.energy != 0.0]
                else:
                    self.wavelen = np.nan
                    self.wavevec = 0.0
            else:
                self.wavelen = np.sqrt(h ** 2 / (2. * m_n * self.energy / JOULES_TO_MEV)) * 1.e10
                self.wavevec = 2. * np.pi / self.wavelength

            self.vel = np.sqrt(2. * self.energy / JOULES_TO_MEV / m_n)
            self.temp = self.energy / k / JOULES_TO_MEV
            self.freq = (self.energy / JOULES_TO_MEV / hbar / 2. / np.pi / 1.e12)

        except AttributeError:
            raise AttributeError("""You must define at least one of the \
                                    following: energy, wavelength, velocity, \
                                    wavevector, temperature, frequency""")

    @property
    def energy(self):
        r"""Energy of the neutron in meV"""
        return self.en

    @energy.setter
    def energy(self, value):
        self._update_values(energy=value)

    @property
    def wavelength(self):
        r"""Wavelength of the neutron in Å"""
        return self.wavelen

    @wavelength.setter
    def wavelength(self, value):
        self._update_values(wavelength=value)

    @property
    def wavevector(self):
        u"""Wavevector k of the neutron in 1/Å"""
        return self.wavevec

    @wavevector.setter
    def wavevector(self, value):
        self._update_values(wavevector=value)

    @property
    def temperature(self):
        r"""Temperature of the neutron in Kelvin"""
        return self.temp

    @temperature.setter
    def temperature(self, value):
        self._update_values(temperature=value)

    @property
    def frequency(self):
        r"""Frequency of the neutron in THz"""
        return self.freq

    @frequency.setter
    def frequency(self, value):
        self._update_values(frequency=value)

    @property
    def velocity(self):
        r"""Velocity of the neutron in m/s"""
        return self.vel

    @velocity.setter
    def velocity(self, value):
        self._update_values(velocity=value)

    @property
    def values(self):
        r"""Prints all of the properties of the Neutron beam

        Parameters
        ----------
        None

        Returns
        -------
        values : string
            A string containing all the properties of the neutron including
            respective units
        """
        values = [u'',
                  u'Energy: {0:3.3f} meV'.format(self.energy),
                  u'Wavelength: {0:3.3f} Å'.format(self.wavelength),
                  u'Wavevector: {0:3.3f} 1/Å'.format(self.wavevector),
                  u'Velocity: {0:3.3f} m/s'.format(self.velocity),
                  u'Temperature: {0:3.3f} K'.format(self.temperature),
                  u'Frequency: {0:3.3f} THz'.format(self.frequency),
                  u'']

        return '\n'.join(values)
