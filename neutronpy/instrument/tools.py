# -*- coding: utf-8 -*-
import numpy as np
from ..crystal import Sample


class _dummy():
    r'''Empty class for constructing empty objects monitor, guide, and detector
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _scalar(v1, v2, lattice):
    r'''Calculates the _scalar product of two vectors, defined by their
    fractional cell coordinates or Miller indexes.

    Parameters
    ----------
    v1 : array
        First input vector

    v2 : array
        Second input vector

    lattice : Sample class
        Class containing unit cell parameters

    Returns
    -------
    s : _scalar
        The _scalar product of the two input vectors scaled by the lattice
        parameters.

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''

    [x1, y1, z1] = v1
    [x2, y2, z2] = v2

    s = x1 * x2 * lattice.a ** 2 + y1 * y2 * lattice.b ** 2 + z1 * z2 * lattice.c ** 2 + \
        (x1 * y2 + x2 * y1) * lattice.a * lattice.b * np.cos(lattice.gamma) + \
        (x1 * z2 + x2 * z1) * lattice.a * lattice.c * np.cos(lattice.beta) + \
        (z1 * y2 + z2 * y1) * lattice.c * lattice.b * np.cos(lattice.alpha)

    return s


def _star(lattice):
    r'''Given lattice parametrs, calculate unit cell volume V, reciprocal
    volume Vstar, and reciprocal lattice parameters.

    Parameters
    ----------
    lattice : Class
        Sample class with the lattice parameters

    Returns
    -------
    [V, Vstar, latticestar] : [float, float, class]
        Returns the unit cell volume, reciprocal cell volume, and a Sample
        Class with reciprocal lattice parameters

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    V = 2 * lattice.a * lattice.b * lattice.c * \
        np.sqrt(np.sin((lattice.alpha + lattice.beta + lattice.gamma) / 2) *
                np.sin((-lattice.alpha + lattice.beta + lattice.gamma) / 2) *
                np.sin((lattice.alpha - lattice.beta + lattice.gamma) / 2) *
                np.sin((lattice.alpha + lattice.beta - lattice.gamma) / 2))

    Vstar = (2 * np.pi) ** 3 / V

    latticestar = Sample(0, 0, 0, 0, 0, 0)
    latticestar.a = 2 * np.pi * lattice.b * lattice.c * np.sin(lattice.alpha) / V
    latticestar.b = 2 * np.pi * lattice.a * lattice.c * np.sin(lattice.beta) / V
    latticestar.c = 2 * np.pi * lattice.b * lattice.a * np.sin(lattice.gamma) / V
    latticestar.alpha = np.arccos((np.cos(lattice.beta) * np.cos(lattice.gamma) -
                                   np.cos(lattice.alpha)) / (np.sin(lattice.beta) * np.sin(lattice.gamma)))
    latticestar.beta = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.gamma) -
                                  np.cos(lattice.beta)) / (np.sin(lattice.alpha) * np.sin(lattice.gamma)))
    latticestar.gamma = np.arccos((np.cos(lattice.alpha) * np.cos(lattice.beta) -
                                   np.cos(lattice.gamma)) / (np.sin(lattice.alpha) * np.sin(lattice.beta)))

    return [V, Vstar, latticestar]


def _modvec(v, lattice):
    r'''Calculates the modulus of a vector, defined by its fractional cell
    coordinates or Miller indexes.

    Parameters
    ----------
    v : array
        Input vector

    lattice : Sample class
        Class containing unit cell parameters

    Returns
    -------
    v : float
        Modulus of the input vector scaled by the sample lattice

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
    Oak Ridge National Laboratory

    '''

    return np.sqrt(_scalar(v, v, lattice))


def GetTau(x, getlabel=False):
    u'''τ-values for common monochromator and analyzer crystals.

    Parameters
    ----------
    x : float or string
        Either the numerical Tau value, in Å\ :sup:`-1`, or a
        common monochromater / analyzer type. Currently included crystals and
        their corresponding τ values are

            +------------------+--------------+-----------+
            | String           |     τ        |           |
            +==================+==============+===========+
            | Be(002)          | 3.50702      |           |
            +------------------+--------------+-----------+
            | Co0.92Fe0.08(200)| 3.54782      | (Heusler) |
            +------------------+--------------+-----------+
            | Cu(002)          | 3.47714      |           |
            +------------------+--------------+-----------+
            | Cu(111)          | 2.99913      |           |
            +------------------+--------------+-----------+
            | Cu(220)          | 4.91642      |           |
            +------------------+--------------+-----------+
            | Cu2MnAl(111)     | 1.82810      | (Heusler) |
            +------------------+--------------+-----------+
            | Ge(111)          | 1.92366      |           |
            +------------------+--------------+-----------+
            | Ge(220)          | 3.14131      |           |
            +------------------+--------------+-----------+
            | Ge(311)          | 3.68351      |           |
            +------------------+--------------+-----------+
            | Ge(511)          | 5.76968      |           |
            +------------------+--------------+-----------+
            | Ge(533)          | 7.28063      |           |
            +------------------+--------------+-----------+
            | PG(002)          | 1.87325      |           |
            +------------------+--------------+-----------+
            | PG(004)          | 3.74650      |           |
            +------------------+--------------+-----------+
            | PG(110)          | 5.49806      |           |
            +------------------+--------------+-----------+
            | Si(111)          | 2.00421      |           |
            +------------------+--------------+-----------+


    getlabel : boolean
        If True, return the name of the common crystal type that is a
        match to the input τ.

    Returns
    -------
    tau : float or string
        Returns either the numerical τ for a given crystal type or the
        name of a crystal type

    Notes
    -----
    Tau is defined as :math:`\\tau = 2\\pi/d`, where d is the d-spacing of the
    crystal in Angstroms.

    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
    Oak Ridge National Laboratory

    '''
    choices = {'pg(002)'.lower(): 1.87325,
               'pg(004)'.lower(): 3.74650,
               'ge(111)'.lower(): 1.92366,
               'ge(220)'.lower(): 3.14131,
               'ge(311)'.lower(): 3.68351,
               'be(002)'.lower(): 3.50702,
               'pg(110)'.lower(): 5.49806,
               'Cu2MnAl(111)'.lower(): 2 * np.pi / 3.437,
               'Co0.92Fe0.08(200)'.lower(): 2 * np.pi / 1.771,
               'Ge(511)'.lower(): 2 * np.pi / 1.089,
               'Ge(533)'.lower(): 2 * np.pi / 0.863,
               'Si(111)'.lower(): 2 * np.pi / 3.135,
               'Cu(111)'.lower(): 2 * np.pi / 2.087,
               'Cu(002)'.lower(): 2 * np.pi / 1.807,
               'Cu(220)'.lower(): 2 * np.pi / 1.278,
               'Cu(111)'.lower(): 2 * np.pi / 2.095}

    if getlabel:
        # return the index/label of the closest monochromator
        choices_ = dict((key, np.abs(value - x)) for (key, value) in choices.items())
        index = min(choices_, key=choices_.get)
        if np.abs(choices_[index]) < 5e-4:
            return index  # the label
        else:
            return ''
    elif isinstance(x, (int, float)):
        return x
    else:
        try:
            return choices[x.lower()]
        except KeyError:
            raise KeyError('Invalid monochromator crystal type.')


def _CleanArgs(*varargin):
    r'''Reshapes input arguments to be row-vectors. N is the length of the
    longest input argument. If any input arguments are shorter than N, their
    first values are replicated to produce vectors of length N. In any case,
    output arguments are row-vectors of length N.

    Parameters
    ----------
    varargin : tuple
        Converts arrays into formats appropriate for the calculation and
        extends arrays that are too short

    Returns
    -------
    [length, varargout] : [int, tuple]
        Returns the length of the input vectors and a tuple containing the
        cleaned vectors

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007,
    Oak Ridge National Laboratory

    '''
    varargout = []
    lengths = np.array([], dtype=np.int32)
    for arg in varargin:
        if type(arg) != list and not isinstance(arg, np.ndarray):
            arg = [arg]
        varargout.append(np.array(arg))
        lengths = np.concatenate((lengths, [len(arg)]))

    length = max(lengths)
    bad = np.where(lengths < length)
    if len(bad[0]) > 0:
        for i in bad[0]:
            varargout[i] = np.concatenate((varargout[i], [varargout[i][-1]] * int(length - lengths[i])))
            lengths[i] = len(varargout[i])

    if len(np.where(lengths < length)[0]) > 0:
        raise ValueError('Fatal error: All inputs must have the same lengths.')

    return [length] + varargout


def _voigt(x, a):
    def _approx1(t):
        return (t * 0.5641896) / (0.5 + t ** 2)

    def _approx2(t, u):
        return (t * (1.410474 + u * 0.5641896)) / (0.75 + (u * (3. + u)))

    def _approx3(t):
        return (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t)))) \
            / (16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t)))))

    def _approx4(t, u):
        return (t * (36183.31 - u * (3321.99 - u * (1540.787 - u * (219.031 - u * (35.7668 - u * (1.320522 - u * 0.56419)))))) \
                / (32066.6 - u * (24322.8 - u * (9022.23 - u * (2186.18 - u * (364.219 - u * (61.5704 - u * (1.84144 - u))))))))

    nx = x.size
    if len(a) == 1:
        a = np.ones(nx, dtype=np.complex64) * a
    y = np.zeros(nx, dtype=np.complex64)

    t = a - 1j * x
    ax = np.abs(x)
    s = ax + a
    u = t ** 2

    good = np.where(a == 0)
    y[good] = np.exp(-x[good] ** 2)

    good = np.where((a >= 15) | (s >= 15))
    y[good] = _approx1(t[good])

    good = np.where((s < 15) & (a < 15) & (a >= 5.5))
    y[good] = _approx2(t[good], u[good])

    good = np.where((s < 15) & (s >= 5.5) & (a < 5.5))
    y[good] = _approx2(t[good], u[good])

    good = np.where((s < 5.5) & (a < 5.5) & (a >= 0.75))
    y[good] = _approx3(t[good])

    good = np.where((s < 5.5) & (a >= 0.195 * ax - 0.176) & (a < 0.75))
    y[good] = _approx3(t[good])

    good = np.where((~((s < 5.5) & (a >= 0.195 * ax - 0.176))) & (a < 0.75))
    y[good] = np.exp(u[good]) - _approx4(t[good], u[good])

    y = np.real(y)
    return y


def project_into_plane(index, r0, rm):
    r'''Projects out-of-plane resolution into a specified plane by performing
    a gaussian integral over the third axis.

    Parameters
    ----------
    index : int
        Index of the axis that should be integrated out

    r0 : float
        Resolution prefactor

    rm : ndarray
        Resolution array

    Returns
    -------
    mp : ndarray
        Resolution matrix in a specified plane

    '''

    r = np.sqrt(2 * np.pi / rm[index, index]) * r0
    mp = rm

    b = rm[:, index] + rm[index, :].T
    b = np.delete(b, index, 0)

    mp = np.delete(mp, index, 0)
    mp = np.delete(mp, index, 1)

    mp -= 1 / (4. * rm[index, index]) * np.outer(b, b.T)

    return [r, mp]


def ellipse(saxis1, saxis2, phi=0, origin=None, npts=31):
    r'''Returns an ellipse.

    Parameters
    ----------
    saxis1 : float
        First semiaxis

    saxis2 : float
        Second semiaxis

    phi : float, optional
        Angle that semiaxes are rotated

    origin : list of floats, optional
        Origin position [x0, y0]

    npts: float, optional
        Number of points in the output arrays.

    Returns
    -------
    [x, y] : list of ndarray
        Two one dimensional arrays representing an ellipse
    '''

    if origin is None:
        origin = [0., 0.]

    theta = np.linspace(0., 2. * np.pi, npts)

    x = np.array(saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi)) + origin[0]
    y = np.array(saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi)) + origin[1]
    return np.vstack((x, y))


def get_bragg_widths(RM):
    bragg = np.array([np.sqrt(8 * np.log(2)) / np.sqrt(RM[0, 0]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[1, 1]),
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[2, 2]),
                      get_phonon_width(0, RM, [0, 0, 0, 1])[1],
                      np.sqrt(8 * np.log(2)) / np.sqrt(RM[3, 3])])

    return bragg * 2


def get_phonon_width(r0, M, C):
    T = np.diag(np.ones(4))
    T[3, :] = np.array(C)
    S = np.matrix(np.linalg.inv(T))
    MP = S.H * M * S
    [rp, MP] = project_into_plane(0, r0, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    [rp, MP] = project_into_plane(0, rp, MP)
    fwhm = np.sqrt(8 * np.log(2)) / np.sqrt(MP[0, 0])

    return [rp, fwhm]


def fproject(mat, i):
    if i == 0:
        v = 2
        j = 1
    if i == 1:
        v = 0
        j = 2
    if i == 2:
        v = 0
        j = 1
    [a, b, c] = mat.shape
    proj = np.zeros((2, 2, c))
    proj[0, 0, :] = mat[i, i, :] - mat[i, v, :] ** 2 / mat[v, v, :]
    proj[0, 1, :] = mat[i, j, :] - mat[i, v, :] * mat[j, v, :] / mat[v, v, :]
    proj[1, 0, :] = mat[j, i, :] - mat[j, v, :] * mat[i, v, :] / mat[v, v, :]
    proj[1, 1, :] = mat[j, j, :] - mat[j, v, :] ** 2 / mat[v, v, :]
    hwhm = proj[0, 0, :] - proj[0, 1, :] ** 2 / proj[1, 1, :]
    hwhm = np.sqrt(2. * np.log(2.)) / np.sqrt(hwhm)

    return hwhm
