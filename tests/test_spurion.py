# -*- coding: utf-8 -*-
r"""Tests spurion search

"""
import warnings

import pytest
from mock import patch
from neutronpy import Instrument, spurion


@patch('sys.stdout')
def test_aluminum_rings(mock_stdout):
    """Check aluminum ring finder
    """
    try:
        spurion.aluminum()
    except:
        pytest.fail('Aluminum ring finder failed')


def test_currat_axe():
    with warnings.catch_warnings(record=True) as w:
        spurion.currat_axe_peaks(Instrument(), [[0.8, 0.8, 0], [1.2, 1.2, 0], 17], [[1, 1, 0]], angle_tol=1)
        assert (len(w) == 3)


if __name__ == "__main__":
    pytest.main()
