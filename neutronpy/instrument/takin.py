r"""Classes to deal with the Takin resolution calculation engine

"""
import os
import pipes
import sys

from .tas_instrument import TripleAxisInstrument
from .tof_instrument import TimeOfFlightInstrument


class TakinTripleAxis(TripleAxisInstrument):
    r"""Interface between TripleAxisInstrument and Takin engine
    """

    def __init__(self, *args, **kwargs):
        super(TakinTripleAxis, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "Instrument('tas', engine='takin')"

    def calc_resolution(self, hkle):
        pass

    def calc_projections(self, hkle):
        pass

    def resolution_convolution(self):
        pass


class TakinTimeOfFlight(TimeOfFlightInstrument):
    r"""Interface between TimeOfFlightInstrument and Takin engine
    """

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "Instrument('tof', engine='takin')"

    def calc_resolution(self, hkle):
        pass

    def calc_projections(self, hkle):
        pass

    def resolution_convolution(self):
        pass
