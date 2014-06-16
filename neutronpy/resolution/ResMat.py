'''
Created on Jun 10, 2014

@author: davidfobes
'''

import numpy as np
from scipy.linalg import block_diag as blkdiag


class Lattice():
    '''
    classdocs
    '''

    def __init__(self, a, b, c, alpha, beta, gamma):
        '''
        Constructor
        '''
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


def scalar(x1, y1, z1, x2, y2, z2, lattice):
    '''
    function s=scalar(x1,y1,z1,x2,y2,z2,lattice)
    #==========================================================================
    # function s=scalarx1,y1,z1,x2,y2,z2,lattice)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  Calculates the scalar product of two vectors, defined by their
    #  fractional cell coordinates or Miller indexes.
    #
    #
    # A. Zheludev, 1999-2007
    # Oak Ridge National Laboratory
    #==========================================================================
    '''

    s = x1 * x2 * lattice.a ** 2 + y1 * y2 * lattice.b ** 2 + z1 * z2 * lattice.c ** 2 + \
        (x1 * y2 + x2 * y1) * lattice.a * lattice.b * np.cos(lattice.gamma) + \
        (x1 * z2 + x2 * z1) * lattice.a * lattice.c * np.cos(lattice.beta) + \
        (z1 * y2 + z2 * y1) * lattice.c * lattice.b * np.cos(lattice.alpha)

    return s


def star(lattice):
    '''
    function [V,Vstar,latticestar]=star(lattice)
    #==========================================================================
    #  function [V,Vst,latticestar]=star(lattice)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  Given lattice parametrs, calculate unit cell volume V, reciprocal volume
    #  Vstar, and reciprocal lattice parameters.
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    V = 2 * lattice.a * lattice.b * lattice.c * \
        np.sqrt(np.sin((lattice.alpha + lattice.beta + lattice.gamma) / 2) * \
        np.sin((-lattice.alpha + lattice.beta + lattice.gamma) / 2) * \
        np.sin((lattice.alpha - lattice.beta + lattice.gamma) / 2) * \
        np.sin((lattice.alpha + lattice.beta - lattice.gamma) / 2))

    Vstar = (2 * np.pi) ** 3 / V

    latticestar = Lattice(0, 0, 0, 0, 0, 0)
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


def modvec(x, y, z, lattice):
    '''
    function m=modvec(x,y,z,lattice)
    #==========================================================================
    # function m=modvec(x,y,z,lattice)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  Calculates the modulus of a vector, defined by its fractional cell
    #  coordinates or Miller indexes.
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    return np.sqrt(scalar(x, y, z, x, y, z, lattice))


def GetLattice(EXP):
    '''
    function [lattice,rlattice]=GetLattice(EXP)
    #==========================================================================
    #  function [lattice,rlattice]=GetLattice(EXP)
    #  Extracts lattice parameters from EXP and returns the direct and
    #  reciprocal lattice parameters in the form used by scalar.m, star.m,etc.
    #  This function is part of ResLib v.3.4
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    s = np.array([item.sample for item in EXP])
    lattice = Lattice(np.array([item.a for item in s]),
                      np.array([item.b for item in s]),
                      np.array([item.c for item in s]),
                      np.array([item.alpha for item in s]) * np.pi / 180,
                      np.array([item.beta for item in s]) * np.pi / 180,
                      np.array([item.gamma for item in s]) * np.pi / 180)
    V, Vstar, rlattice = star(lattice)  # @UnusedVariable

    return [lattice, rlattice]


def GetTau(x, getlabel=False):
    '''
    function tau = GetTau(x, getlabel)
    #==========================================================================
    #  function GetTau(tau)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  Tau-values for common monochromator crystals
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    choices = {'pg(002)'.lower(): 1.87325,
               'pg(004)'.lower(): 3.74650,
               'ge(111)'.lower(): 1.92366,
               'ge(220)'.lower(): 3.14131,
               'ge(311)'.lower(): 3.68351,
               'be(002)'.lower(): 3.50702,
               'pg(110)'.lower(): 5.49806,
               'Cu2MnAl(111)'.lower(): 2 * np.pi / 3.437,
               'Co0.92Fe0.08'.lower(): 2 * np.pi / 1.771,
               'Ge(511)'.lower(): 2 * np.pi / 1.089,
               'Ge(533)'.lower(): 2 * np.pi / 0.863,
               'Si(111)'.lower(): 2 * np.pi / 3.135,
               'Cu(111)'.lower(): 2 * np.pi / 2.087,
               'Cu(002)'.lower(): 2 * np.pi / 1.807,
               'Cu(220)'.lower(): 2 * np.pi / 1.278,
               'Cu(111)'.lower(): 2 * np.pi / 2.095}

    if getlabel:
        # return the index/label of the closest monochromator
        choices_ = {key: np.abs(value - x) for key, value in choices.items}
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


def CleanArgs(*varargin):
    '''
    function [len,varargout]=CleanArgs(varargin)
    #==========================================================================
    #  function [N,X,Y,Z,..]=CleanArgs(X,Y,Z,..)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  Reshapes input arguments to be row-vectors. N is the length of the
    #  longest input argument. If any input arguments are shorter than N, their
    #  first values are replicated to produce vectors of length N. In any case,
    #  output arguments are row-vectors of length N.
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
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


def ResMat(Q, W, EXP):
    '''
    function [R0,RM]=ResMat(Q,W,EXP)
    #==========================================================================
    #  function [R0,RM]=ResMat(Q,W,EXP)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  For a momentum transfer Q and energy transfers W,
    #  given experimental conditions specified in EXP,
    #  calculates the Cooper-Nathans resolution matrix RM and
    #  Cooper-Nathans Resolution prefactor R0.
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    # 0.424660900144 = FWHM2RMS
    # CONVERT1=0.4246609*pi/60/180
    CONVERT1 = np.pi / 60. / 180.  # TODO: FIX constant from CN. 0.4246
    CONVERT2 = 2.072

    [length, Q, W, EXP] = CleanArgs(Q, W, EXP)

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
        #----------------------------------------------------------------------
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
        #----------------------------------------------------------------------
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
        #------------------------------------------------------------------
        # Redefine sample geometry
        psi = thetas - phi  # Angle from sample geometry X axis to Q
        rot = np.matrix(np.zeros((3, 3), dtype=np.float64))
        rot[0, 0] = np.cos(psi)
        rot[1, 1] = np.cos(psi)
        rot[0, 1] = np.sin(psi)
        rot[1, 0] = -np.sin(psi)
        rot[2, 2] = 1.
        # sshape=rot'*sshape*rot
        sshape = rot * np.matrix(sshape) * rot.T

        #-------------------------------------------------------------------
        # Definition of matrix G
        G = 1. / np.array([alpha[0], alpha[1], beta[0], beta[1], alpha[2], alpha[3], beta[2], beta[3]], dtype=np.float64) ** 2
        G = np.matrix(np.diag(G))
        #----------------------------------------------------------------------
        # Definition of matrix F
        F = 1. / np.array([etam, etamv, etaa, etaav], dtype=np.float64) ** 2
        F = np.matrix(np.diag(F))

        #-------------------------------------------------------------------
        # Definition of matrix A
        A[0, 0] = ki / 2. / np.tan(thetam)
        A[0, 1] = -A[0, 0]
        A[3, 4] = kf / 2. / np.tan(thetaa)
        A[3, 5] = -A[3, 4]
        A[1, 1] = ki
        A[2, 3] = ki
        A[4, 4] = kf
        A[5, 6] = kf

        #-------------------------------------------------------------------
        # Definition of matrix C
        C[0, 0] = 1. / 2.
        C[0, 1] = 1. / 2.
        C[2, 4] = 1. / 2.
        C[2, 5] = 1. / 2.
        C[1, 2] = 1. / (2. * np.sin(thetam))
        C[1, 3] = -C[1, 2]  # mistake in paper
        C[3, 6] = 1. / (2. * np.sin(thetaa))
        C[3, 7] = -C[3, 6]

        #-------------------------------------------------------------------
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
        #----------------------------------------------------------------------
        # Definition of matrix S
        Sinv = np.matrix(blkdiag(np.array(bshape, dtype=np.float32), mshape, sshape, ashape, dshape))  # S-1 matrix
        S = Sinv.I
        #----------------------------------------------------------------------
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
        #-------------------------------------------------------------------
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

        #----------------------------------------------------------------------
        # Definition of resolution matrix M
        if method == 1 or method == 'Popovici':
            K = S + T.T * F * T
            H = np.linalg.inv(D * np.linalg.inv(K) * D.T)
            Ninv = A * np.linalg.inv(H + G) * A.T  # Popovici Eq 20
        else:
            H = G + C.T * F * C  # Popovici Eq 8
            Ninv = A * np.linalg.inv(H) * A.T  # Cooper-Nathans (in Popovici Eq 10)
            # Horizontally focusing analyzer if needed
            if horifoc > 0:
                Ninv = np.linalg.inv(Ninv)
                Ninv[4, 4] = (1. / (kf * alpha[3])) ** 2
                Ninv[4, 3] = 0.
                Ninv[3, 4] = 0.
                Ninv[3, 3] = (np.tan(thetaa) / (etaa * kf)) ** 2
                Ninv = np.linalg.inv(Ninv)

        Minv = B * Ninv * B.T  # Popovici Eq 3

        # TODO: FIX added factor 5.545 from ResCal5
        M = 8. * np.log(2.) * np.linalg.inv(Minv)
            # Correction factor 8*log(2) as input parameters
            # are expressed as FWHM.

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
        #-------------------------------------------------------------------
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

        #-------------------------------------------------------------------

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
            d[2, 1] = D(1, 1)
            d[2, 4] = -D(1, 1)
            d[1, 2] = np.cos(thetam) / L1mon
            d[1, 3] = np.sin(thetam) / L1mon
            d[1, 5] = 0.
            d[1, 6] = 1. / L1mon
            d[3, 4] = -1. / L1mon
            if method == 1 or method == 'Popovici':
                # Popovici
                Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * \
                        np.sqrt(np.linalg.det(f) / np.linalg.det((np.matrix(d) * \
                        (np.matrix(s) + np.matrix(t).T * np.matrix(f) * np.matrix(t)).I * np.matrix(d).T).I + np.matrix(g)))
            else:
                # Cooper-Nathans
                Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * \
                        np.sqrt(np.linalg.det(f) / np.linalg.det(np.matrix(g) + \
                        np.matrix(c).T * np.matrix(f) * np.matrix(c)))

            R0_ = R0_ / Rmon
            R0_ = R0_ * ki  # 1/ki monitor efficiency

        #----------------------------------------------------------------------
        # Transform prefactor to Chesser-Axe normalization
        R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(M))
        #----------------------------------------------------------------------
        # Include kf/ki part of cross section
        R0_ = R0_ * kf / ki
        #----------------------------------------------------------------------
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

        #----------------------------------------------------------------------
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

        #----------------------------------------------------------------------
        print('R0_:{}'.format(R0_))
        print('R0:{}'.format(R0))
        R0[ind] = R0_
        RM[:, :, ind] = M[:, :]

    return [R0, RM]


def StandardSystem(EXP):
    '''
    function [x,y,z,lattice,rlattice]=StandardSystem(EXP)
    #==========================================================================
    #  function [x,y,z,lattice,rlattice]=StandardSystem(EXP)
    #  This function is part of ResLib v.3.4
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    [lattice, rlattice] = GetLattice(EXP)
    length = len(EXP)

    orient1 = np.zeros((3, length), dtype=np.float64)
    orient2 = np.zeros((3, length), dtype=np.float64)
    for i in range(length):
        orient1[:, i] = EXP[i].orient1
        orient2[:, i] = EXP[i].orient2

    modx = modvec(orient1[0, :], orient1[1, :], orient1[2, :], rlattice)

    x = orient1
    x[0, :] = x[0, :] / modx  # First unit basis vector
    x[1, :] = x[1, :] / modx
    x[2, :] = x[2, :] / modx

    proj = scalar(orient2[0, :], orient2[1, :], orient2[2, :], x[0, :], x[1, :], x[2, :], rlattice)

    y = orient2
    y[0, :] = y[0, :] - x[0, :] * proj
    y[1, :] = y[1, :] - x[1, :] * proj
    y[2, :] = y[2, :] - x[2, :] * proj

    mody = modvec(y[0, :], y[1, :], y[2, :], rlattice)

    if len(np.where(mody <= 0)[0]) > 0:
        raise ValueError('??? Fatal error: Orienting vectors are colinear!')

    y[0, :] = y[0, :] / mody  # Second unit basis vector
    y[1, :] = y[1, :] / mody
    y[2, :] = y[2, :] / mody

    z = np.zeros((3, length), dtype=np.float64)
    z[0, :] = x[1, :] * y[2, :] - y[1, :] * x[2, :]
    z[1, :] = x[2, :] * y[0, :] - y[2, :] * x[0, :]
    z[2, :] = -x[1, :] * y[0, :] + y[1, :] * x[0, :]

    proj = scalar(z[0, :], z[1, :], z[2, :], x[0, :], x[1, :], x[2, :], rlattice)

    z[0, :] = z[0, :] - x[0, :] * proj
    z[1, :] = z[1, :] - x[1, :] * proj
    z[2, :] = z[2, :] - x[2, :] * proj

    proj = scalar(z[0, :], z[1, :], z[2, :], y[0, :], y[1, :], y[2, :], rlattice)

    z[0, :] = z[0, :] - y[0, :] * proj
    z[1, :] = z[1, :] - y[1, :] * proj
    z[2, :] = z[2, :] - y[2, :] * proj

    modz = modvec(z[0, :], z[1, :], z[2, :], rlattice)

    z[0, :] = z[0, :] / modz  # Third unit basis vector
    z[1, :] = z[1, :] / modz
    z[2, :] = z[2, :] / modz

    return [x, y, z, lattice, rlattice]


def res_calc(H, K, L, W, EXP):
    '''
    function [R0,RMS, RM]=ResMatS(H,K,L,W,EXP)
    #==========================================================================
    #  function [R0,RMS]=ResMatS(H,K,L,W,EXP)
    #  ResLib v.3.4
    #==========================================================================
    #
    #  For a scattering vector (H,K,L) and  energy transfers W,
    #  given experimental conditions specified in EXP,
    #  calculates the Cooper-Nathans resolution matrix RMS and
    #  Cooper-Nathans Resolution prefactor R0 in a coordinate system
    #  defined by the crystallographic axes of the sample.
    #
    # A. Zheludev, 1999-2006
    # Oak Ridge National Laboratory
    #==========================================================================
    '''
    [length, H, K, L, W, EXP] = CleanArgs(H, K, L, W, EXP)
    [x, y, z, sample, rsample] = StandardSystem(EXP)  # @UnusedVariable

    Q = modvec(H, K, L, rsample)
    uq = np.zeros((3, length), dtype=np.float64)
    uq[0, :] = H / Q  # Unit vector along Q
    uq[1, :] = K / Q
    uq[2, :] = L / Q

    xq = scalar(x[0, :], x[1, :], x[2, :], uq[0, :], uq[1, :], uq[2, :], rsample)
    yq = scalar(y[0, :], y[1, :], y[2, :], uq[0, :], uq[1, :], uq[2, :], rsample)
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
        RMS[:, :, i] = (np.matrix(tmat[:, :, i])).T * np.matrix(RM[:, :, i]) * np.matrix(tmat[:, :, i])

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
    return [R0, RMS, RM]
