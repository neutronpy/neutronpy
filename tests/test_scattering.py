# -*- coding: utf-8 -*-
r"""Unit tests for scattering functions

"""
import pytest
from neutronpy import Data, scattering


def test_gen_hkl():
    pattern = scattering.pattern.HKLGenerator()
    pattern.generate_hkl_positions()


def test_find_equiv_pos():
    pattern = scattering.pattern.HKLGenerator()
    pattern.find_equivalent_positions()


def test_app_rules():
    pattern = scattering.pattern.HKLGenerator()
    pattern.apply_scattering_rules()


def test_multiplicity():
    pattern = scattering.pattern.HKLGenerator()
    pattern.find_site_multiplicity()


def test_polarization_correction():
    data = Data()
    scattering.polarization.polarization_correction(data, data, data, data)


if __name__ == "__main__":
    pytest.main()
