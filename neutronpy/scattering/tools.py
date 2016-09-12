# -*- coding: utf-8 -*-
r"""Tools for visualizing the scattering geometry

"""

import numpy as np
try:
    from math import gcd
except ImportError:
    from fractions import gcd


def vector_gcd(vector):
    vector = np.array(vector)

    gcd_value = 1
    for i, element in enumerate(vector[:-1]):
        gcd_temp = gcd(element, vector[i + 1])

        if i == 0:
            gcd_value = gcd_temp
        else:
            gcd_value = gcd(gcd_temp, gcd_value)

    if gcd_value == 0:
        return vector

    return vector / gcd_value


def get_coords_in_reciprocal_space(vector, u, v):
    r"""

    Parameters
    ----------
    vector
    u
    v

    Returns
    -------

    """
    return [np.dot(vector, u) / np.linalg.norm(u) ** 2, np.dot(vector, v) / np.linalg.norm(v) ** 2,
            np.dot(vector, np.cross(u, v)) / np.linalg.norm(np.cross(u, v)) ** 2]


def get_vector_in_reciprocal_units(vector, abc):
    r"""

    Parameters
    ----------
    vector
    abc

    Returns
    -------

    """
    return np.array(vector) * 2 * np.pi / np.array(abc)
