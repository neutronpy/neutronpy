# -*- coding: utf-8 -*-
"""Testing for Energy conversions

"""
import pytest
import numpy as np
from neutronpy import Energy


def test_energy():
    """Test that the output string is correct
    """
    energy = Energy(energy=25.)
    assert (np.round(energy.energy) == 25.0)
    assert (np.round(energy.wavelength, 4) == 1.8089)
    assert (np.round(energy.wavevector, 3) == 3.473)
    assert (np.round(energy.velocity, 3) == 2186.967)
    assert (np.round(energy.temperature, 3) == 290.113)
    assert (np.round(energy.frequency, 3) == 6.045)

    string_test = u"\nEnergy: {0:3.3f} meV"
    string_test += u"\nWavelength: {1:3.3f} Å"
    string_test += u"\nWavevector: {2:3.3f} 1/Å"
    string_test += u"\nVelocity: {3:3.3f} m/s"
    string_test += u"\nTemperature: {4:3.3f} K"
    string_test += u"\nFrequency: {5:3.3f} THz\n"
    string_test = string_test.format(25.0, 1.8089, 3.473, 2186.967,
                                     290.113, 6.045)

    assert (energy.values == string_test)


def test_energy_setters():
    """Tests the energy setters
    """
    e = Energy(energy=25.)

    e.energy = 25
    assert (np.round(e.wavelength, 4) == 1.8089)

    e.wavevector = 3.5
    assert (np.round(e.energy, 1) == 25.4)

    e.velocity = 2180
    assert (np.round(e.energy, 1) == 24.8)

    e.temperature = 290
    assert (np.round(e.energy, 1) == 25.0)

    e.frequency = 6
    assert (np.round(e.energy, 1) == 24.8)

    e.wavelength = 1.9
    assert (np.round(e.energy, 1) == 22.7)


if __name__ == "__main__":
    pytest.main()
