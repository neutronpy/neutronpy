# -*- coding: utf-8 -*-
r"""Tests lattice math

"""
import numpy as np
import pytest
from neutronpy.crystal import lattice
from neutronpy.crystal.exceptions import LatticeError

unitcell = lattice.Lattice(4, 4, 4, 90, 90, 90)


def test_get_angle_between_planes():
    """Tests get angle between planes defined by two vectors
    """
    assert (unitcell.get_angle_between_planes([1, 0, 0], [1, 1, 1]) - 54.73561031724535 < 1e-6)


def test_get_d_spacing():
    """Tests d-spacing for given HKL
    """
    assert (unitcell.get_d_spacing([1, 1, 1]) == unitcell.a / np.sqrt(3))


def test_get_q():
    """Tests q for given HKL
    """
    assert (abs(unitcell.get_q([1, 1, 1]) - 2 * np.pi / unitcell.b * np.sqrt(3)) < 1e-12)


def test_get_two_theta():
    """Tests 2theta for given HKL
    """
    assert (unitcell.get_two_theta([1, 1, 1], 2) == 51.317812546510552)


def test_constants():
    """Test that gettters/setters work properly
    """
    abg = np.array([unitcell.alpha, unitcell.beta, unitcell.gamma])
    abc = np.array([unitcell.a, unitcell.b, unitcell.c])
    abg_rad = np.array([unitcell.alpha_rad, unitcell.beta_rad, unitcell.gamma_rad])
    abgstar_rad = np.array([unitcell.alphastar_rad, unitcell.betastar_rad, unitcell.gammastar_rad])
    abgstar = np.array([unitcell.alphastar, unitcell.betastar, unitcell.gammastar])
    abcstar = np.array([unitcell.astar, unitcell.bstar, unitcell.cstar])

    assert (np.all(unitcell.abc == abc))
    assert (np.all(unitcell.abg == abg))
    assert (np.all(unitcell.abg_rad == abg_rad))
    assert (np.all(unitcell.abg_rad == np.deg2rad(abg)))
    assert (np.round(unitcell.volume, 12) == 4 ** 3)
    assert (np.round(unitcell.reciprocal_volume, 12) == np.round(8 * np.pi ** 3 / (4 ** 3), 12))
    assert (unitcell.astar == unitcell.b * unitcell.c * np.sin(unitcell.alpha_rad) / unitcell.volume * 2 * np.pi)
    assert (unitcell.bstar == unitcell.a * unitcell.c * np.sin(unitcell.beta_rad) / unitcell.volume * 2 * np.pi)
    assert (unitcell.cstar == unitcell.a * unitcell.b * np.sin(unitcell.gamma_rad) / unitcell.volume * 2 * np.pi)
    assert (np.all(abgstar_rad == np.deg2rad(abgstar)))
    assert (np.all(unitcell.reciprocal_abc == abcstar))
    assert (np.all(unitcell.reciprocal_abg == abgstar))
    assert (np.all(unitcell.reciprocal_abg_rad == abgstar_rad))
    assert (np.all(np.round(unitcell.Bmatrix * unitcell.Bmatrix.T, 12) == np.round(unitcell.Gstar, 12)))


def test_lattice_type():
    """Test lattice type determination
    """
    test_cell = unitcell
    assert (test_cell.lattice_type == 'cubic')

    test_cell = lattice.Lattice(1, 1, 2, 90, 90, 90)
    assert (test_cell.lattice_type == 'tetragonal')

    test_cell = lattice.Lattice(1, 2, 3, 90, 90, 90)
    assert (test_cell.lattice_type == 'orthorhombic')

    test_cell = lattice.Lattice(1, 2, 3, 90, 89, 90)
    assert (test_cell.lattice_type == 'monoclinic')

    test_cell = lattice.Lattice(1, 1, 1, 39, 39, 39)
    assert (test_cell.lattice_type == 'rhombohedral')

    test_cell = lattice.Lattice(1, 1, 1, 39, 39, 39)
    assert (test_cell.lattice_type == 'rhombohedral')

    test_cell = lattice.Lattice(1, 1, 2, 90, 90, 120)
    assert (test_cell.lattice_type == 'hexagonal')

    test_cell = lattice.Lattice(1, 2, 3, 30, 60, 120)
    assert (test_cell.lattice_type == 'triclinic')

    test_cell = lattice.Lattice(1, 1, 2, 90, 90, 150)
    with pytest.raises(LatticeError):
        getattr(test_cell, 'lattice_type')


def test_goniometer_constants():
    """Test constants
    """
    pass


if __name__ == "__main__":
    pytest.main()
