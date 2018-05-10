# -*- coding: utf-8 -*-
"""Custom Exceptions for the instrument module

"""


class DetectorError(Exception):
    """For Exceptions related to the properties of the Detector
    """
    pass


class ScatteringTriangleError(Exception):
    """For Exceptions related to the Scattering Triangle not closing
    """
    pass


class MonochromatorError(Exception):
    """For Exceptions related to the Monochromator properties
    """
    pass


class AnalyzerError(Exception):
    """For Exceptions related to the Monochromator properties
    """
    pass


class ChopperError(Exception):
    """For Exceptions related to the Chopper properties
    """
    pass


class GoniometerError(Exception):
    """For Exceptions related to the Goniometer positions
    """
    pass


class InstrumentError(Exception):
    """For general, unclassified types of Instrument Exceptions
    """
    pass
