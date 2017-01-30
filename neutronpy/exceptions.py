# -*- coding: utf-8 -*-
r"""Defines custom exception handling for neutronpy

"""


class EnergyError(Exception):
    """Error passed by Energy objects
    """
    pass


class ModelError(Exception):
    """Error passed by model functions
    """
    pass


class SpurionCalculationError(Exception):
    """Error in the calculation of Spurions
    """
    pass