# -*- coding: utf-8 -*-
r"""Tools for Fitter class
"""
from functools import wraps

import numpy as np


def convert_params(params):
    r"""

    Parameters
    ----------
    params : obj, list
        Will convert a LMFIT Parameters object or pass parameters list into a
        list of parameters.

    Returns
    -------

    """
    if not isinstance(params, (list, np.ndarray)):
        parvals = params.valuesdict()
        parin = [parvals['p{0}'.format(i)] for i in range(len(parvals))]
    else:
        parin = params

    return parin


def residual_wrapper(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        parin = convert_params(args[0])
        argin = (parin,) + args[1:]
        return function(*argin, **kwargs)
    return wrapper
