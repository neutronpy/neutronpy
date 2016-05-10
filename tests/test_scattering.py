# -*- coding: utf-8 -*-
r"""Unit tests for scattering functions

"""
import unittest
from neutronpy import scattering, Data


class PatternTest(unittest.TestCase):
    r"""Unit tests for Powder Pattern calculations
    """

    def test_gen_hkl(self):
        pattern = scattering.pattern.HKLGenerator()
        pattern.generate_hkl_positions()

    def test_find_equiv_pos(self):
        pattern = scattering.pattern.HKLGenerator()
        pattern.find_equivalent_positions()

    def test_app_rules(self):
        pattern = scattering.pattern.HKLGenerator()
        pattern.apply_scattering_rules()

    def test_multiplicity(self):
        pattern = scattering.pattern.HKLGenerator()
        pattern.find_site_multiplicity()


class PolarizationTest(unittest.TestCase):
    r"""Unit tests for Polarization tools
    """

    def test_polarization_correction(self):
        data = Data()
        scattering.polarization.polarization_correction(data, data, data, data)


if __name__ == "__main__":
    unittest.main()
