r"""Testing of the resolution library - TOF

"""
import numpy as np
import pytest
from neutronpy import instrument
from neutronpy.instrument.exceptions import *

# TODO: Add bragg width tests
# TODO: Add resolution volume tests

def test_calc_res():
    """Test calculation of known problem
    """

    instr = instrument.Instrument(instrument_type="tof")

    instr.l_pm = 1567
    instr.l_ms = 150
    instr.l_sd = 350

    instr.theta_i = 0.
    instr.phi_i = 0.

    instr.sigma_l_pm = 12.5
    instr.sigma_l_ms = 2.
    instr.sigma_l_sd = 0.75

    instr.sigma_theta_i = .459
    instr.sigma_theta = .1
    instr.sigma_phi_i = .688
    instr.sigma_phi = .2

    instr.tau_p = 66.59
    instr.tau_m = 16.65
    instr.tau_d = .1

    instr.detector.shape = "cylindrical"
    instr.detector.orientation = "vertical"

    instr.sample.a = 5
    instr.sample.b = 5
    instr.sample.c = 5
    instr.sample.u = [1, 0, 0]
    instr.sample.v = [0, 1, 0]

    instr.ei.wavevector = 1.13333333

    instr.calc_resolution([1, 1, 0, 0])

    RMS = np.matrix([[ 15863.63, -8809.27,  0,        1797.56],
                     [-8809.27,   52131.95, 0,       -4760.37],
                     [ 0,         0,        7862.15,  0      ],
                     [ 1797.56,  -4760.37,  0,        681.03 ]])

    RM = np.matrix([[ 15950.82, 11483.59, 0,       -1667.17],
                    [ 11483.59, 27107.89, 0,       -3690.13],
                    [ 0,        0,        4978.76,  0      ],
                    [-1667.17, -3690.13,  0,        681.03]])

    R0 = 2.099e-5

    assert (np.all(np.round(instr.RMS, 2) == np.round(RMS, 2)))
    assert (np.all(np.round(instr.RM, 2) == np.round(RM, 2)))
    assert (instr.R0 - R0 < 1e-8)

def test_calc_res_cases():
    """Test calculation of various cases
    """

    instr = instrument.Instrument(instrument_type="tof")
    instr.detector.shape = "spherical"
    instr.calc_resolution([1, 0, 0, 0])

    instr = instrument.Instrument(instrument_type="tof")
    instr.calc_resolution([1, 0, 0, 0])
