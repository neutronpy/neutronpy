# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import block_diag as blkdiag


class Sample():
    u'''Private class containing sample information.

    Parameters
    ----------
    a : float
        Unit cell length in angstroms

    b : float
        Unit cell length in angstroms

    c : float
        Unit cell length in angstroms

    alpha : float
        Angle between b and c in degrees

    beta : float
        Angle between a and c in degrees

    gamma : float
        Angle between a and b in degrees

    mosaic : float, optional
        Sample mosaic (FWHM) in arc minutes

    direct : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)

    Returns
    -------
    Sample : object

    '''
    def __init__(self, a, b, c, alpha, beta, gamma, mosaic=60, direct=1):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mosaic = mosaic
        self.dir = direct


class _Monochromator():
    u'''Private class containing monochromator information.

    Parameters
    ----------
    tau : float or string
        Tau value for the monochromator (or analyzer)

    mosaic : int
        Mosaic of the crystal in arc minutes

    dir : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively)

    Returns
    -------
    Monochromator : class

    '''
    def __init__(self, tau, mosaic, direct=-1):
        self.tau = tau
        self.mosaic = mosaic
        self.dir = direct


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
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''

    return np.sqrt(_scalar(v, v, lattice))


def _GetLattice(EXP):
    r'''Extracts lattice parameters from EXP and returns the direct and
    reciprocal lattice parameters in the form used by _scalar.m, _star.m, etc.

    Parameters
    ----------
    EXP : Instrument Class
        The Instrument information

    Returns
    -------
    [lattice, rlattice] : [class, class]
        Returns the direct and reciprocal lattice sample classes

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    s = np.array([item.sample for item in EXP])
    lattice = Sample(np.array([item.a for item in s]),
                      np.array([item.b for item in s]),
                      np.array([item.c for item in s]),
                      np.array([item.alpha for item in s]) * np.pi / 180,
                      np.array([item.beta for item in s]) * np.pi / 180,
                      np.array([item.gamma for item in s]) * np.pi / 180)
    V, Vstar, rlattice = _star(lattice)  # @UnusedVariable

    return [lattice, rlattice]


def GetTau(x, getlabel=False):
    u'''τ-values for common monochromator and analyzer crystals.

    Parameters
    ----------
    x : float or string
        Either the numerical Tau value, in Å\ :sup:`-1`, or a
        common monochromater / analyzer type. Currently included crystals and their
        corresponding τ values are

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
    Tau is defined as :math:`\tau = 2\pi/d`, where d is the d-spacing of the crystal in Angstroms.

    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

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
        choices_ = dict((key, np.abs(value - x)) for (key, value) in choices.items)
        index = min(choices_, key=choices_.get)
        if choices[index] < 5e-4:
            tau = choices[index]  # the label
        else:
            tau = ''
    elif isinstance(x, (int, float)):
        tau = x
    else:
        try:
            tau = choices[x.lower()]
        except ValueError:
            raise ValueError('Invalid monochromator crystal type.')

    return tau


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
        Returns the length of the input vectors and a tuple containing the cleaned vectors

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

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


def _StandardSystem(EXP):
    r'''Returns rotation matrices to calculate resolution in the sample view
    instead of the instrument view

    Parameters
    ----------
    EXP : class
        Instrument class

    Returns
    -------
    [x, y, z, lattice, rlattice] : [ndarray, ndarray, ndarray, class, class]
        Returns the rotation matrices and real and reciprocal lattice sample classes

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    [lattice, rlattice] = _GetLattice(EXP)
    length = len(EXP)

    orient1 = np.zeros((3, length), dtype=np.float64)
    orient2 = np.zeros((3, length), dtype=np.float64)
    for i in range(length):
        orient1[:, i] = EXP[i].orient1
        orient2[:, i] = EXP[i].orient2

    modx = _modvec([orient1[0, :], orient1[1, :], orient1[2, :]], rlattice)

    x = orient1
    x[0, :] = x[0, :] / modx  # First unit basis vector
    x[1, :] = x[1, :] / modx
    x[2, :] = x[2, :] / modx

    proj = _scalar([orient2[0, :], orient2[1, :], orient2[2, :]], [x[0, :], x[1, :], x[2, :]], rlattice)

    y = orient2
    y[0, :] = y[0, :] - x[0, :] * proj
    y[1, :] = y[1, :] - x[1, :] * proj
    y[2, :] = y[2, :] - x[2, :] * proj

    mody = _modvec([y[0, :], y[1, :], y[2, :]], rlattice)

    if len(np.where(mody <= 0)[0]) > 0:
        raise ValueError('??? Fatal error: Orienting vectors are colinear!')

    y[0, :] = y[0, :] / mody  # Second unit basis vector
    y[1, :] = y[1, :] / mody
    y[2, :] = y[2, :] / mody

    z = np.zeros((3, length), dtype=np.float64)
    z[0, :] = x[1, :] * y[2, :] - y[1, :] * x[2, :]
    z[1, :] = x[2, :] * y[0, :] - y[2, :] * x[0, :]
    z[2, :] = -x[1, :] * y[0, :] + y[1, :] * x[0, :]

    proj = _scalar([z[0, :], z[1, :], z[2, :]], [x[0, :], x[1, :], x[2, :]], rlattice)

    z[0, :] = z[0, :] - x[0, :] * proj
    z[1, :] = z[1, :] - x[1, :] * proj
    z[2, :] = z[2, :] - x[2, :] * proj

    proj = _scalar([z[0, :], z[1, :], z[2, :]], [y[0, :], y[1, :], y[2, :]], rlattice)

    z[0, :] = z[0, :] - y[0, :] * proj
    z[1, :] = z[1, :] - y[1, :] * proj
    z[2, :] = z[2, :] - y[2, :] * proj

    modz = _modvec([z[0, :], z[1, :], z[2, :]], rlattice)

    z[0, :] = z[0, :] / modz  # Third unit basis vector
    z[1, :] = z[1, :] / modz
    z[2, :] = z[2, :] / modz

    return [x, y, z, lattice, rlattice]


def ResMat(Q, W, EXP):
    r'''For a momentum transfer Q and energy transfers W, given experimental
    conditions specified in EXP, calculates the Cooper-Nathans resolution
    matrix RM and Cooper-Nathans Resolution prefactor R0.

    Parameters
    ----------
    Q : ndarray or list of ndarray
        The Q vectors in reciprocal space at which resolution should be calculated

    W : float or list of floats
        The energy transfers at which resolution should be calculated

    EXP : Class
        Instrument class containing the relevant information about the instrument

    Returns
    -------
    [R0, RM] : list(float, ndarray)
        Resolution pre-factor (R0) and resolution matrix (RM) at the given reciprocal
        lattice vectors and energy transfers

    Notes
    -----
    Translated from ResLib 3.4c, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

    '''
    # 0.424660900144 = FWHM2RMS
    # CONVERT1=0.4246609*pi/60/180
    CONVERT1 = np.pi / 60. / 180.  # TODO: FIX constant from CN. 0.4246
    CONVERT2 = 2.072

    [length, Q, W, EXP] = _CleanArgs(Q, W, EXP)

    RM = np.zeros((4, 4, length), dtype=np.float64)
    R0 = np.zeros(length, dtype=np.float64)
    RM_ = np.zeros((4, 4), dtype=np.float64)  # @UnusedVariable
    D = np.matrix(np.zeros((8, 13), dtype=np.float64))
    d = np.matrix(np.zeros((4, 7), dtype=np.float64))
    T = np.matrix(np.zeros((4, 13), dtype=np.float64))
    t = np.matrix(np.zeros((2, 7), dtype=np.float64))
    A = np.matrix(np.zeros((6, 8), dtype=np.float64))
    C = np.matrix(np.zeros((4, 8), dtype=np.float64))
    B = np.matrix(np.zeros((4, 6), dtype=np.float64))
    for ind in range(length):
        # the method to use
        method = 0
        if hasattr(EXP[ind], 'method'):
            method = EXP[ind].method

        # Assign default values and decode parameters
        moncor = 0
        if hasattr(EXP[ind], 'moncor'):
            moncor = EXP[ind].moncor

        alpha = EXP[ind].hcol * CONVERT1
        beta = EXP[ind].vcol * CONVERT1
        mono = EXP[ind].mono
        etam = mono.mosaic * CONVERT1
        etamv = etam
        if hasattr(mono, 'vmosaic') and (method == 1 or method == 'Popovici'):
            etamv = mono.vmosaic * CONVERT1

        ana = EXP[ind].ana
        etaa = ana.mosaic * CONVERT1
        etaav = etaa
        if hasattr(ana, 'vmosaic'):
            etaav = ana.vmosaic * CONVERT1

        sample = EXP[ind].sample
        infin = -1
        if hasattr(EXP[ind], 'infin'):
            infin = EXP[ind].infin

        efixed = EXP[ind].efixed
        epm = 1  # @UnusedVariable
        if hasattr(EXP[ind], 'dir1'):
            epm = EXP[ind].dir1  # @UnusedVariable

        ep = 1  # @UnusedVariable
        if hasattr(EXP[ind], 'dir2'):
            ep = EXP[ind].dir2  # @UnusedVariable

        monitorw = 1.
        monitorh = 1.
        beamw = 1.
        beamh = 1.
        monow = 1.
        monoh = 1.
        monod = 1.
        anaw = 1.
        anah = 1.
        anad = 1.
        detectorw = 1.
        detectorh = 1.
        sshape = np.identity(3)
        L0 = 1.
        L1 = 1.
        L1mon = 1.
        L2 = 1.
        L3 = 1.
        monorv = 1.e6
        monorh = 1.e6
        anarv = 1.e6
        anarh = 1.e6
        if hasattr(EXP[ind], 'beam'):
            beam = EXP[ind].beam
            if hasattr(beam, 'width'):
                beamw = beam.width ** 2 / 12.

            if hasattr(beam, 'height'):
                beamh = beam.height ** 2 / 12.

        bshape = np.diag([beamw, beamh])
        if hasattr(EXP[ind], 'monitor'):
            monitor = EXP[ind].monitor
            if hasattr(monitor, 'width'):
                monitorw = monitor.width ** 2 / 12.

            monitorh = monitorw
            if hasattr(monitor, 'height'):
                monitorh = monitor.height ** 2 / 12.

        monitorshape = np.diag([monitorw, monitorh])
        if hasattr(EXP[ind], 'detector'):
            detector = EXP[ind].detector
            if hasattr(detector, 'width'):
                detectorw = detector.width ** 2 / 12.

            if hasattr(detector, 'height'):
                detectorh = detector.height ** 2 / 12.

        dshape = np.diag([detectorw, detectorh])
        if hasattr(mono, 'width'):
            monow = mono.width ** 2 / 12.

        if hasattr(mono, 'height'):
            monoh = mono.height ** 2 / 12.

        if hasattr(mono, 'depth'):
            monod = mono.depth ** 2 / 12.

        mshape = np.diag([monod, monow, monoh])
        if hasattr(ana, 'width'):
            anaw = ana.width ** 2 / 12.

        if hasattr(ana, 'height'):
            anah = ana.height ** 2 / 12.

        if hasattr(ana, 'depth'):
            anad = ana.depth ** 2 / 12.

        ashape = np.diag([anad, anaw, anah])
        if hasattr(sample, 'width') and hasattr(sample, 'depth') and hasattr(sample, 'height'):
            sshape = np.diag([sample.depth, sample.width, sample.height]) ** 2 / 12.
        elif hasattr(sample, 'shape'):
            sshape = sample.shape / 12.

        if hasattr(EXP[ind], 'arms'):
            arms = EXP[ind].arms
            L0 = arms[0]
            L1 = arms[1]
            L2 = arms[2]
            L3 = arms[3]
            L1mon = L1
            if len(arms) > 4:
                L1mon = arms[4]

        if hasattr(mono, 'rv'):
            monorv = mono.rv

        if hasattr(mono, 'rh'):
            monorh = mono.rh

        if hasattr(ana, 'rv'):
            anarv = ana.rv

        if hasattr(ana, 'rh'):
            anarh = ana.rh

        taum = GetTau(mono.tau)
        taua = GetTau(ana.tau)

        horifoc = -1
        if hasattr(EXP[ind], 'horifoc'):
            horifoc = EXP[ind].horifoc

        if horifoc == 1:
            alpha[2] = alpha[2] * np.sqrt(8 * np.log(2) / 12.)

        em = 1  # @UnusedVariable
        if hasattr(EXP[ind], 'mondir'):
            em = EXP[ind].mondir  # @UnusedVariable

        sm = EXP[ind].mono.dir
        ss = EXP[ind].sample.dir
        sa = EXP[ind].ana.dir

        # Calculate angles and energies
        w = W[ind]
        q = Q[ind]
        ei = efixed
        ef = efixed
        if infin > 0:
            ef = efixed - w
        else:
            ei = efixed + w
        ki = np.sqrt(ei / CONVERT2)
        kf = np.sqrt(ef / CONVERT2)

        thetam = np.arcsin(taum / (2. * ki)) * sm  # added sign(em) K.P.
        thetaa = np.arcsin(taua / (2. * kf)) * sa
        s2theta = np.arccos((ki ** 2 + kf ** 2 - q ** 2) / (2. * ki * kf)) * ss  # 2theta sample @IgnorePep8
        if np.iscomplex(s2theta):
            raise ValueError(': KI,KF,Q triangle will not close (kinematic equations). Change the value of KFIX,FX,QH,QK or QL.')

        thetas = s2theta / 2.
        phi = np.arctan2(-kf * np.sin(s2theta), ki - kf * np.cos(s2theta))

        # Redefine sample geometry
        psi = thetas - phi  # Angle from sample geometry X axis to Q
        rot = np.matrix(np.zeros((3, 3), dtype=np.float64))
        rot[0, 0] = np.cos(psi)
        rot[1, 1] = np.cos(psi)
        rot[0, 1] = np.sin(psi)
        rot[1, 0] = -np.sin(psi)
        rot[2, 2] = 1.

        # sshape=rot'*sshape*rot
        sshape = rot * np.matrix(sshape) * rot.H

        # Definition of matrix G
        G = 1. / np.array([alpha[0], alpha[1], beta[0], beta[1], alpha[2], alpha[3], beta[2], beta[3]], dtype=np.float64) ** 2
        G = np.matrix(np.diag(G))

        # Definition of matrix F
        F = 1. / np.array([etam, etamv, etaa, etaav], dtype=np.float64) ** 2
        F = np.matrix(np.diag(F))

        # Definition of matrix A
        A[0, 0] = ki / 2. / np.tan(thetam)
        A[0, 1] = -A[0, 0]
        A[3, 4] = kf / 2. / np.tan(thetaa)
        A[3, 5] = -A[3, 4]
        A[1, 1] = ki
        A[2, 3] = ki
        A[4, 4] = kf
        A[5, 6] = kf

        # Definition of matrix C
        C[0, 0] = 1. / 2.
        C[0, 1] = 1. / 2.
        C[2, 4] = 1. / 2.
        C[2, 5] = 1. / 2.
        C[1, 2] = 1. / (2. * np.sin(thetam))
        C[1, 3] = -C[1, 2]  # mistake in paper
        C[3, 6] = 1. / (2. * np.sin(thetaa))
        C[3, 7] = -C[3, 6]

        # Definition of matrix B
        B[0, 0] = np.cos(phi)
        B[0, 1] = np.sin(phi)
        B[0, 3] = -np.cos(phi - s2theta)
        B[0, 4] = -np.sin(phi - s2theta)
        B[1, 0] = -B[0, 1]
        B[1, 1] = B[0, 0]
        B[1, 3] = -B[0, 4]
        B[1, 4] = B[0, 3]
        B[2, 2] = 1.
        B[2, 5] = -1.
        B[3, 0] = 2. * CONVERT2 * ki
        B[3, 3] = -2. * CONVERT2 * kf

        # Definition of matrix S
        Sinv = np.matrix(blkdiag(np.array(bshape, dtype=np.float32), mshape, sshape, ashape, dshape))  # S-1 matrix
        S = Sinv.I

        # Definition of matrix T
        T[0, 0] = -1. / (2. * L0)  # mistake in paper
        T[0, 2] = np.cos(thetam) * (1. / L1 - 1. / L0) / 2.
        T[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1 - 2. * (monorh * np.sin(thetam))) / 2.
        T[0, 5] = np.sin(thetas) / (2. * L1)
        T[0, 6] = np.cos(thetas) / (2. * L1)
        T[1, 1] = -1. / (2. * L0 * np.sin(thetam))
        T[1, 4] = (1. / L0 + 1. / L1 - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
        T[1, 7] = -1. / (2. * L1 * np.sin(thetam))
        T[2, 5] = np.sin(thetas) / (2. * L2)
        T[2, 6] = -np.cos(thetas) / (2. * L2)
        T[2, 8] = np.cos(thetaa) * (1. / L3 - 1. / L2) / 2.
        T[2, 9] = np.sin(thetaa) * (1. / L2 + 1. / L3 - 2. * (anarh * np.sin(thetaa))) / 2.
        T[2, 11] = 1. / (2. * L3)
        T[3, 7] = -1. / (2. * L2 * np.sin(thetaa))
        T[3, 10] = (1. / L2 + 1. / L3 - 2. * np.sin(thetaa) / anarv) / (2. * np.sin(thetaa))
        T[3, 12] = -1. / (2. * L3 * np.sin(thetaa))

        # Definition of matrix D
        # Lots of index mistakes in paper for matrix D
        D[0, 0] = -1. / L0
        D[0, 2] = -np.cos(thetam) / L0
        D[0, 3] = np.sin(thetam) / L0
        D[2, 1] = D[0, 0]
        D[2, 4] = -D[0, 0]
        D[1, 2] = np.cos(thetam) / L1
        D[1, 3] = np.sin(thetam) / L1
        D[1, 5] = np.sin(thetas) / L1
        D[1, 6] = np.cos(thetas) / L1
        D[3, 4] = -1. / L1
        D[3, 7] = -D[3, 4]
        D[4, 5] = np.sin(thetas) / L2
        D[4, 6] = -np.cos(thetas) / L2
        D[4, 8] = -np.cos(thetaa) / L2
        D[4, 9] = np.sin(thetaa) / L2
        D[6, 7] = -1. / L2
        D[6, 10] = -D[6, 7]
        D[5, 8] = np.cos(thetaa) / L3
        D[5, 9] = np.sin(thetaa) / L3
        D[5, 11] = 1. / L3
        D[7, 10] = -D[5, 11]
        D[7, 12] = D[5, 11]

        # Definition of resolution matrix M
        if method == 1 or method == 'Popovici':
            K = S + T.H * F * T
            H = np.linalg.inv(D * np.linalg.inv(K) * D.H)
            Ninv = A * np.linalg.inv(H + G) * A.H  # Popovici Eq 20
        else:
            H = G + C.H * F * C  # Popovici Eq 8
            Ninv = A * np.linalg.inv(H) * A.H  # Cooper-Nathans (in Popovici Eq 10)
            # Horizontally focusing analyzer if needed
            if horifoc > 0:
                Ninv = np.linalg.inv(Ninv)
                Ninv[4, 4] = (1. / (kf * alpha[3])) ** 2
                Ninv[4, 3] = 0.
                Ninv[3, 4] = 0.
                Ninv[3, 3] = (np.tan(thetaa) / (etaa * kf)) ** 2
                Ninv = np.linalg.inv(Ninv)

        Minv = B * Ninv * B.H  # Popovici Eq 3

        # TODO: FIX added factor 5.545 from ResCal5
        M = 8. * np.log(2.) * np.linalg.inv(Minv)
        # Correction factor 8*log(2) as input parameters are expressed as FWHM.

        # TODO: rows-columns 3-4 swapped for ResPlot to work.
        # Inactivate as we want M=[x,y,z,E]
#         RM_[0, 0] = M[0, 0]
#         RM_[1, 0] = M[1, 0]
#         RM_[0, 1] = M[0, 1]
#         RM_[1, 1] = M[1, 1]
#
#         RM_[0, 2] = M[0, 3]
#         RM_[2, 0] = M[3, 0]
#         RM_[2, 2] = M[3, 3]
#         RM_[2, 1] = M[3, 1]
#         RM_[1, 2] = M[1, 3]
#
#         RM_[0, 3] = M[0, 2]
#         RM_[3, 0] = M[2, 0]
#         RM_[3, 3] = M[2, 2]
#         RM_[3, 1] = M[2, 1]
#         RM_[1, 3] = M[1, 2]

        # Calculation of prefactor, normalized to source
        Rm = ki ** 3 / np.tan(thetam)
        Ra = kf ** 3 / np.tan(thetaa)
        R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * np.sin(thetam) * np.sin(thetaa))

        if method == 1 or method == 'Popovici':
            # Popovici
            R0_ = R0_ * np.sqrt(np.linalg.det(F) / np.linalg.det(np.matrix(H) + np.matrix(G)))
        else:
            # Cooper-Nathans (popovici Eq 5 and 9)
            R0_ = R0_ * np.sqrt(np.linalg.det(np.matrix(F)) / np.linalg.det(np.matrix(H)))

        # Normalization to flux on monitor
        if moncor == 1:
            g = G[0:3, 0:3]
            f = F[0:1, 0:1]
            c = C[0:1, 0:3]
            t[0, 0] = -1. / (2. * L0)  # mistake in paper
            t[0, 2] = np.cos(thetam) * (1. / L1mon - 1. / L0) / 2.
            t[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1mon - 2. / (monorh * np.sin(thetam))) / 2.
            t[0, 6] = 1. / (2. * L1mon)
            t[1, 1] = -1. / (2. * L0 * np.sin(thetam))
            t[1, 4] = (1. / L0 + 1. / L1mon - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
            sinv = blkdiag(np.array(bshape, dtype=np.float32), mshape, monitorshape)  # S-1 matrix
            s = np.matrix(sinv).I
            d[0, 0] = -1. / L0
            d[0, 2] = -np.cos(thetam) / L0
            d[0, 3] = np.sin(thetam) / L0
            d[2, 1] = D[1, 1]
            d[2, 4] = -D[1, 1]
            d[1, 2] = np.cos(thetam) / L1mon
            d[1, 3] = np.sin(thetam) / L1mon
            d[1, 5] = 0.
            d[1, 6] = 1. / L1mon
            d[3, 4] = -1. / L1mon
            if method == 1 or method == 'Popovici':
                # Popovici
                Rmon = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) *
                        np.sqrt(np.linalg.det(f) / np.linalg.det((np.matrix(d) *
                                                                  (np.matrix(s) + np.matrix(t).H * np.matrix(f) * np.matrix(t)).I *
                                                                  np.matrix(d).H).I + np.matrix(g))))
            else:
                # Cooper-Nathans
                Rmon = (Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) *
                        np.sqrt(np.linalg.det(f) / np.linalg.det(np.matrix(g) + np.matrix(c).H * np.matrix(f) * np.matrix(c))))

            R0_ = R0_ / Rmon
            R0_ = R0_ * ki  # 1/ki monitor efficiency

        # Transform prefactor to Chesser-Axe normalization
        R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(M))

        # Include kf/ki part of cross section
        R0_ = R0_ * kf / ki

        # Take care of sample mosaic if needed
        # [S. A. Werner & R. Pynn, J. Appl. Phys. 42, 4736, (1971), eq 19]
        if hasattr(sample, 'mosaic'):
            etas = sample.mosaic * CONVERT1
            etasv = etas
            if hasattr(sample, 'vmosaic'):
                etasv = sample.vmosaic * CONVERT1

            # TODO: FIX changed RM_(4,4) and M(4,4) to M(3,3)
            R0_ = R0_ / np.sqrt((1. + (q * etasv) ** 2 * M[2, 2]) * (1. + (q * etas) ** 2 * M[1, 1]))
            # Minv=RM_^(-1)
            Minv[1, 1] = Minv[1, 1] + q ** 2 * etas ** 2
            Minv[2, 2] = Minv[2, 2] + q ** 2 * etasv ** 2
            # TODO: FIX add 8*log(2) factor for mosaicities in FWHM
            M = 8 * np.log(2) * np.linalg.inv(Minv)

        # Take care of analyzer reflectivity if needed [I. Zaliznyak, BNL]
        if hasattr(ana, 'thickness') and hasattr(ana, 'Q'):
            KQ = ana.Q
            KT = ana.thickness
            toa = (taua / 2.) / np.sqrt(kf ** 2 - (taua / 2.) ** 2)
            smallest = alpha[3]
            if alpha[3] > alpha[2]:
                smallest = alpha[2]
            Qdsint = KQ * toa
            dth = (np.arange(1, 201) / 200.) * np.sqrt(2. * np.log(2.)) * smallest
            wdth = np.exp(-dth ** 2 / 2. / etaa ** 2)
            sdth = KT * Qdsint * wdth / etaa / np.sqrt(2. * np.pi)
            rdth = 1. / (1 + 1. / sdth)
            reflec = sum(rdth) / sum(wdth)
            R0_ = R0_ * reflec

        R0[ind] = R0_
        RM[:, :, ind] = M[:, :]

    return [R0, RM]


def project_into_plane(rm, index):
    r'''Projects out-of-plane resolution into a specified plane by performing
    a gaussian integral over the third axis.

    Parameters
    ----------
    rm : ndarray
        Resolution array

    index : int
        Index of the axis that should be integrated out

    Returns
    -------
    mp : ndarray
        Resolution matrix in a specified plane

    '''

    mp = rm

    b = rm[:, index] + rm[index, :].T
    b = np.delete(b, index, 0)

    mp = np.delete(mp, index, 0)
    mp = np.delete(mp, index, 1)

    mp -= 1 / (4. * rm[index, index]) * np.outer(b, b.T)

    return mp


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

    x = np.array(saxis1 * np.cos(theta) * np.cos(phi) - saxis2 * np.sin(theta) * np.sin(phi) + origin[0])
    y = np.array(saxis1 * np.cos(theta) * np.sin(phi) + saxis2 * np.sin(theta) * np.cos(phi) + origin[1])
    return np.vstack((x, y))


class Instrument(object):
    u'''An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    efixed : float
        Fixed energy, either ei or ef, depending on the instrument configuration.

    sample : obj
        Sample lattice constants, parameters, mosaic, and orientation (reciprocal-space orienting vectors).

    hcol : list(4)
        Horizontal Soller collimations in minutes of arc starting from the neutron guide.

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the neutron guide.

    mono_tau : str or float, optional
        The monochromator reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common monochromator types.

    mono_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc.

    ana_tau : str or float, optional
        The analyzer reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common analyzer types.

    ana_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc.

    Attributes
    ----------
    method
    moncar
    mono
    ana
    hcol
    vcol
    arms
    efixed
    sample
    orient1
    orient2
    dir1
    dir2
    mondir
    infin
    beam
    detector
    monitor
    Smooth

    Methods
    -------
    calc_resolution
    calc_projections
    get_resolution_params
    '''

    def __init__(self, efixed, sample, hcol, vcol=None, arms=None,
                 mono='PG(002)', mono_mosaic=25, ana='PG(002)', ana_mosaic=25,
                 **kwargs):

        if vcol is None:
            vcol = [120, 120, 120, 120]

        if arms is None:
            arms = [150, 150, 150, 150]

        self.mono = _Monochromator(mono, mono_mosaic)
        self.ana = _Monochromator(ana, ana_mosaic)
        self.hcol = np.array(hcol)
        self.vcol = np.array(vcol)
        self.arms = np.array(arms)
        self.efixed = efixed
        self.sample = sample
        self.orient1 = np.array(sample.u)
        self.orient2 = np.array(sample.v)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mono(self):
        u'''A structure that describes the monochromator.
        
        Parameters
        ----------
        tau : str or float 
            The monochromator reciprocal lattice vector in Å\ :sup:`-1`. Instead of
            a numerical input one can use one of the following keyword strings:
            
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
        
        mosaic : int
            The monochromator mosaic in minutes of arc.
        
        vmosaic : int
            The vertical mosaic of monochromator in minutes of arc. If
            this field is left unassigned, an isotropic mosaic is assumed.
        '''
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value

    @property
    def ana(self):
        u'''A structure that describes the analyzer and contains fields as in :attr:`mono`
        plus optional fields.
        
        Parameters
        ----------
        thickness: float 
            The analyzer thickness in cm for ideal-crystal reflectivity
            corrections (Section II C 3). If no reflectivity corrections are to be made, this field
            should remain unassigned or set to a negative value.
        
        Q : float
            The kinematic reflectivity coefficient for this correction. It is given by 
        
            .. math::    Q = \\frac{4|F|^2}{V_0} \\frac{(2\\pi)^3}{\\tau^3}, 
        
            where V0 is the unit cell volume for the analyzer crystal, F is the structure factor of 
            the analyzer reflection, and τ is the analyzer reciprocal lattice vector. 
            For PG(002) Q = 0.1287. Leave this field unassigned or make it negative if you don’t
            want the correction done.
        
        horifoc : bool 
            A flag that is set to 1 if a horizontally focusing analyzer is used
            (Section II D). In this case ``hcol[2]`` (see below) is the angular size of the
            analyzer, as seen from the sample position. If the field is unassigned or equal to
            -1, a flat analyzer is assumed. Note that this option is only available with the
            Cooper-Nathans method.
        '''
        return self._ana

    @ana.setter
    def ana(self, value):
        self._ana = value

    @property
    def method(self):
        '''Selects the computation method. 
        If ``method=0`` or left undefined, a Cooper-Nathans calculation is
        performed. For a Popovici calculation set ``method=1``.
        '''
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def moncar(self):
        '''Selects the type of normalization used to calculate ``R0``. 
        If ``moncor=1`` or left undefined, ``R0`` is calculated in normalization to monitor counts (Section
        II C 2). 1/k\ :sub:`i` monitor efficiency correction is included automatically. To normalize
        ``R0`` to source flux (Section II C 1), use ``moncor=0``.
        '''
        return self._moncar

    @moncar.setter
    def moncar(self, value):
        self._moncar = value

    @property
    def hcol(self):
        ''' The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. In case of a horizontally-focusing
        analyzer ``hcol[2] is the angular size of the analyzer, as seen from the sample
        position. If the beam divergence is limited by a neutron guide, the corresponding
        element of :attr:`hcol` is the negative of the guide’s *m*-value. For example, for a 58-Ni
        guide (*m*=1.2) before the monochromator, `hcol[0]` should be -1.2.
        '''
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        '''The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. If the beam divergence is limited
        by a neutron guide, the corresponding element of :attr:`vcol` is the negative of the
        guide’s *m*-value. For example, for a 58-Ni guide (*m*=1.2) before the monochromator,
        ``vcol[0]`` should be -1.2.
        '''
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        '''distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and monochromator and
        monitor, respectively. The 5th element is only needed if ``moncor=1``
        '''
        return self._arms

    @arms.setter
    def arms(self, value):
        self._arms = value

    @property
    def efixed(self):
        '''the fixed incident or final neutron energy, in meV.
        '''
        return self._efixed

    @efixed.setter
    def efixed(self, value):
        self._efixed = value

    @property
    def sample(self):
        '''A structure that describes the sample. 
        
        It contains the following fields:
        
        * EXP.sample.mosaic is the FWHM sample mosaic in the scattering plane in minutes
        of arc. If this field is left unassigned, no sample mosaic corrections (section
        II E) are performed.

        * ``sample.vmosaic`` is the vertical sample mosaic in minutes of arc. If this field
        is left unassigned, isotropic mosaic is assumed.
        '''
        return self._sample

    @sample.setter
    def sample(self, value):
        self._sample = value

    @property
    def orient1(self):
        '''Miller indexes of the first reciprocal-space orienting vector for
        the S coordinate system, as explained in Section II G.
        '''
        return self._orient1

    @orient1.setter
    def orient1(self, value):
        self._orient1 = value

    @property
    def orient2(self):
        '''Miller indexes of the second reciprocal-space orienting vector
        for the S coordinate system, as explained in Section II G.
        '''
        return self._orient2

    @orient2.setter
    def orient2(self, value):
        self._orient2 = value

    @property
    def dir1(self):
        '''defines the scattering direction in the monochromator. This field is equal
        to 1 or left unassigned if the scattering direction in the monochromator is opposite to
        that in the sample. Set this field to -1 if the sample and monochromator scattering
        directions are the same.
        '''
        return self._dir1

    @dir1.setter
    def dir1(self, value):
        self._dir1 = value

    @property
    def dir2(self):
        '''defines the scattering direction in the analyzer. This field is equal to 1 or
        left unassigned if the scattering direction in the analyzer is opposite to that in the
        sample. Set this field to -1 if the sample and analyzer scattering directions are the
        same.
        '''
        return self._dir2

    @dir2.setter
    def dir2(self, value):
        self._dir2 = value

    @property
    def mondir(self):
        '''defines the scattering angle in the monochromator which is positive
        (counter-clockwise) if this field is absent or positive, and negative (clockwise) otherwise
        [10].
        '''
        return self._mondir

    @mondir.setter
    def mondir(self, value):
        self._mondir = value

    @property
    def infin(self):
        '''a flag set to -1 or left unassigned if the final energy is fixed, or set to +1
        in a fixed-incident setup.
        '''
        return self._infin

    @infin.setter
    def infin(self, value):
        self._infin = value

    @property
    def beam(self):
        '''
        '''
        return self._beam

    @beam.setter
    def beam(self, value):
        self._beam = value

    @property
    def detector(self):
        '''
        '''
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def monitor(self):
        '''
        '''
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def Smooth(self):
        u'''Defines the smoothing parameters as explained in Section II H. Leave this
        field unassigned if you don’t want this correction done.
        
        * ``Smooth.E`` is the smoothing FWHM in energy (meV). A small number means
        “no smoothing along this direction”.
        
        * ``Smooth.X`` is the smoothing FWHM along the first orienting vector (x0 axis)
        in Å\ :sup:`-1`.
        
        * ``Smooth.Y`` is the smoothing FWHM along the y axis in Å\ :sup:`-1`.
        
        * ``Smooth.Z`` is the smoothing FWHM along the vertical direction in Å\ :sup:`-1`.
        '''
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value

    def calc_resolution(self, hkle, npts=36):
        r'''For a scattering vector (H,K,L) and  energy transfers W, given
        experimental conditions specified in EXP, calculates the Cooper-Nathans
        resolution matrix RMS and Cooper-Nathans Resolution prefactor R0 in a
        coordinate system defined by the crystallographic axes of the sample.

        Parameters
        ----------
        hkle : list
            Array of the scattering vector and energy transfer at which the
            calculation should be performed

        npts : int, optional
            Number of points in the ouput curves

        Notes
        -----
            Translated from ResLib, originally authored by A. Zheludev, 1999-2007, Oak Ridge National Laboratory

        '''
        [H, K, L, W] = hkle
        EXP = self

        [length, H, K, L, W, EXP] = _CleanArgs(H, K, L, W, EXP)
        self.H, self.K, self.L, self.W = H, K, L, W

        [x, y, z, sample, rsample] = _StandardSystem(EXP)  # @UnusedVariable

        Q = _modvec([H, K, L], rsample)
        uq = np.zeros((3, length), dtype=np.float64)
        uq[0, :] = H / Q  # Unit vector along Q
        uq[1, :] = K / Q
        uq[2, :] = L / Q

        xq = _scalar([x[0, :], x[1, :], x[2, :]], [uq[0, :], uq[1, :], uq[2, :]], rsample)
        yq = _scalar([y[0, :], y[1, :], y[2, :]], [uq[0, :], uq[1, :], uq[2, :]], rsample)
        zq = 0  # @UnusedVariable # scattering vector assumed to be in (orient1,orient2) plane

        tmat = np.zeros((4, 4, length), dtype=np.float64)  # Coordinate transformation matrix
        tmat[3, 3, :] = 1.
        tmat[2, 2, :] = 1.
        tmat[0, 0, :] = xq
        tmat[0, 1, :] = yq
        tmat[1, 1, :] = xq
        tmat[1, 0, :] = -yq

        RMS = np.zeros((4, 4, length), dtype=np.float64)
        rot = np.zeros((3, 3), dtype=np.float64)
        EXProt = EXP

        # Sample shape matrix in coordinate system defined by scattering vector
        for i in range(length):
            sample = EXP[i].sample
            if hasattr(sample, 'shape'):
                rot[0, 0] = tmat[0, 0, i]
                rot[1, 0] = tmat[1, 0, i]
                rot[0, 1] = tmat[0, 1, i]
                rot[1, 1] = tmat[1, 1, i]
                rot[2, 2] = tmat[2, 2, i]
                EXProt[i].sample.shape = np.matrix(rot) * np.matrix(sample.shape) * np.matrix(rot.T)

        [R0, RM] = ResMat(Q, W, EXProt)

        for i in range(length):
            RMS[:, :, i] = np.matrix(tmat[:, :, i]).T * np.matrix(RM[:, :, i]) * np.matrix(tmat[:, :, i])

        mul = np.zeros((4, 4))
        e = np.identity(4)
        for i in range(length):
            if hasattr(EXP[i], 'Smooth'):
                if EXP[i].Smooth.X:
                    mul[0, 0] = 1 / (EXP[i].Smooth.X ** 2 / 8 / np.log(2))
                    mul[1, 1] = 1 / (EXP[i].Smooth.Y ** 2 / 8 / np.log(2))
                    mul[2, 2] = 1 / (EXP[i].Smooth.E ** 2 / 8 / np.log(2))
                    mul[3, 3] = 1 / (EXP[i].Smooth.Z ** 2 / 8 / np.log(2))
                    R0[i] = R0[i] / np.sqrt(np.linalg.det(e / RMS[:, :, i])) * np.sqrt(np.linalg.det(e / mul + e / RMS[:, :, i]))
                    RMS[:, :, i] = e / (e / mul + e / RMS[:, :, i])

        self.R0, self.RMS, self.RM = R0, RMS, RM
        self.calc_projections([H, K, L, W], npts=npts)

    def calc_projections(self, hkle, npts=36):
        r'''Calculates the resolution ellipses for projections and slices from the resolution matrix.

        Parameters
        ----------
        hkle : list
            Positions at which projections should be calculated.

        npts : int, optional
            Number of points in the outputted ellipse curve

        Returns
        -------
        projections : dictionary
            A dictionary containing projections in the planes: QxQy, QxW, and QyW, both projections and slices

        '''
        try:
            A = np.array(self.RMS)
        except Exception:
            raise Exception('Resolution calculation has not been performed')

        const = 1.17741  # half width factor

        self.projections = {'QxQy': np.zeros((2, npts, A.shape[-1])),
                            'QxQySlice': np.zeros((2, npts, A.shape[-1])),
                            'QxW': np.zeros((2, npts, A.shape[-1])),
                            'QxWSlice': np.zeros((2, npts, A.shape[-1])),
                            'QyW': np.zeros((2, npts, A.shape[-1])),
                            'QyWSlice': np.zeros((2, npts, A.shape[-1]))}

        for ind in range(A.shape[-1]):
            # Remove the vertical component from the matrix.
            B = np.vstack((np.hstack((A[0, :2:1, ind], A[0, 3, ind])),
                           np.hstack((A[1, :2:1, ind], A[1, 3, ind])),
                           np.hstack((A[3, :2:1, ind], A[3, 3, ind]))))

            # Projection into Qx, Qy plane
            MP = project_into_plane(B, 2)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQy'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[1][ind]], npts)

            # Slice through Qx,Qy plane
            MP = np.array(A[:2:1, :2:1, ind])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQySlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[1][ind]], npts)

            # Projection into Qx, W plane

            MP = project_into_plane(B, 1)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[3][ind]], npts)

            # Slice through Qx,W plane
            MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[0][ind], hkle[3][ind]], npts)

            # Projections into Qy, W plane
            MP = project_into_plane(B, 0)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1][ind], hkle[3][ind]], npts)

            # Slice through Qy,W plane
            MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta, [hkle[1][ind], hkle[3][ind]], npts)

    def get_resolution_params(self, hkle, plane, mode='project'):
        r'''Returns parameters for the resolution gaussian.

        Parameters
        ----------
        hkle : list of floats
            Position and energy for which parameters should be returned

        plane : 'QxQy' | 'QxQySlice' | 'QxW' | 'QxWSlice' | 'QyW' | 'QyWSlice'
            Two dimensional plane for which parameters should be returned

        mode : 'project' | 'slice'
            Return the projection into or slice through the chosen plane

        Returns
        -------
        tuple : R0, RMxx, RMyy, RMxy
            Parameters for the resolution gaussian

        '''
        A = self.RMS

        ind = np.where((self.H == hkle[0]) & (self.K == hkle[1]) & (self.L == hkle[2]) & (self.W == hkle[3]))
        if len(ind[0]) == 0:
            raise ValueError('Resolution at provided HKLE has not been calculated.')

        ind = ind[0][0]

        # Remove the vertical component from the matrix
        B = np.vstack((np.hstack((A[0, :2:1, ind], A[0, 3, ind])),
                       np.hstack((A[1, :2:1, ind], A[1, 3, ind])),
                       np.hstack((A[3, :2:1, ind], A[3, 3, ind]))))

        if plane == 'QxQy':
            if mode == 'project':
                # Projection into Qx, Qy plane
                R0 = np.sqrt(2 * np.pi / B[2, 2]) * self.R0[ind]
                MP = project_into_plane(B, 2)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,Qy plane
                MP = np.array(A[:2:1, :2:1, ind])
                return (self.R0[ind], MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QxW':
            if mode == 'project':
                # Projection into Qx, W plane
                R0 = np.sqrt(2 * np.pi / B[1, 1]) * self.R0[ind]
                MP = project_into_plane(B, 1)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,W plane
                MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])
                return (self.R0[ind][0], MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QyW':
            if mode == 'project':
                # Projections into Qy, W plane
                R0 = np.sqrt(2 * np.pi / B[0, 0]) * self.R0
                MP = project_into_plane(B, 0)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qy,W plane
                MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])
                return (self.R0[ind][0], MP[0, 0], MP[1, 1], MP[0, 1])
