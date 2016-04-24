# -*- coding: utf-8 -*-
import numpy as np


def build_Q(args, **kwargs):
    u'''Method for constructing **Q**\ (*q*, ℏω, temp) from h, k, l,
    energy, and temperature

    Parameters
    ----------
    args : dict
        A dictionary of the `h`, `k`, `l`, `e` and `temp` arrays to form into
        a column oriented array

    Returns
    -------
    Q : ndarray
        Returns **Q**\ (h, k, l, e, temp) with shape (N, 5) in a column
        oriented array.

    '''
    return np.vstack((args[i].flatten() for i in
                      ['h', 'k', 'l', 'e', 'temp'])).T
