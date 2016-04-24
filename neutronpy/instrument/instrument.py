# -*- coding: utf-8 -*-
r'''Define an instrument for resolution calculations

'''
import numpy as np
from scipy.linalg import block_diag as blkdiag
from .analyzer import Analyzer
from .monochromator import Monochromator
from .tools import ellipse, project_into_plane, _dummy, _star, _CleanArgs, _modvec, _scalar, GetTau, _voigt
from ..energy import Energy
from .plot import PlotInstrument
from ..crystal import Sample


class Instrument(PlotInstrument):
    u'''An object that represents a Triple Axis Spectrometer (TAS) instrument
    experimental configuration, including a sample.

    Parameters
    ----------
    efixed : float, optional
        Fixed energy, either ei or ef, depending on the instrument
        configuration. Default: 14.7

    sample : obj, optional
        Sample lattice constants, parameters, mosaic, and orientation
        (reciprocal-space orienting vectors). Default: A crystal with
        a,b,c = 6,7,8 and alpha,beta,gamma = 90,90,90 and orientation
        vectors u=[1 0 0] and v=[0 1 0].

    hcol : list(4)
        Horizontal Soller collimations in minutes of arc starting from the
        neutron guide. Default: [40 40 40 40]

    vcol : list(4), optional
        Vertical Soller collimations in minutes of arc starting from the
        neutron guide. Default: [120 120 120 120]

    mono_tau : str or float, optional
        The monochromator reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common monochromator types.
        Default: 'PG(002)'

    mono_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    ana_tau : str or float, optional
        The analyzer reciprocal lattice vector in Å\ :sup:`-1`,
        given either as a float, or as a string for common analyzer types.
        Default: 'PG(002)'

    ana_mosaic : float, optional
        The mosaic of the monochromator in minutes of arc. Default: 25

    Attributes
    ----------
    method
    moncor
    mono
    ana
    hcol
    vcol
    arms
    efixed
    sample
    orient1
    orient2
    infin
    beam
    detector
    monitor
    Smooth
    guide

    Methods
    -------
    calc_resolution
    calc_resolution_in_Q_coords
    calc_projections
    get_angles_and_Q
    get_lattice
    get_resolution_params
    plot_projections
    plot_ellipsoid
    plot_instrument
    resolution_convolution
    resolution_convolution_SMA
    plot_slice
    description_string

    '''
    def __init__(self, efixed=14.7, sample=None, hcol=None, vcol=None, mono='PG(002)',
                 mono_mosaic=25, ana='PG(002)', ana_mosaic=25, **kwargs):

        if sample is None:
            sample = Sample(6, 7, 8, 90, 90, 90)
            sample.u = [1, 0, 0]
            sample.v = [0, 1, 0]

        if hcol is None:
            hcol = [40, 40, 40, 40]

        if vcol is None:
            vcol = [120, 120, 120, 120]

        self.mono = Monochromator(mono, mono_mosaic)
        self.ana = Analyzer(ana, ana_mosaic)
        self.hcol = np.array(hcol)
        self.vcol = np.array(vcol)
        self.efixed = efixed
        self.sample = sample
        self.orient1 = np.array(sample.u)
        self.orient2 = np.array(sample.v)
        self.detector = _dummy()
        self.monitor = _dummy()
        self.guide = _dummy()

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mono(self):
        u'''A structure that describes the monochromator.

        Parameters
        ----------
        tau : str or float
            The monochromator reciprocal lattice vector in Å\ :sup:`-1`.
            Instead of a numerical input one can use one of the following
            keyword strings:

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

        dir : int
            Direction of the crystal (left or right, -1 or +1, respectively).
            Default: -1 (left-handed coordinate frame).

        rh : float
            Horizontal curvature of the monochromator in cm.

        rv : float
            Vertical curvature of the monochromator in cm.

        '''
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value

    @property
    def ana(self):
        u'''A structure that describes the analyzer and contains fields as in
        :attr:`mono` plus optional fields.

        Parameters
        ----------
        thickness: float
            The analyzer thickness in cm for ideal-crystal reflectivity
            corrections (Section II C 3). If no reflectivity corrections are to
            be made, this field should remain unassigned or set to a negative
            value.

        Q : float
            The kinematic reflectivity coefficient for this correction. It is
            given by

            .. math::    Q = \\frac{4|F|**2}{V_0} \\frac{(2\\pi)**3}{\\tau**3},

            where V0 is the unit cell volume for the analyzer crystal, F is the
            structure factor of the analyzer reflection, and τ is the analyzer
            reciprocal lattice vector. For PG(002) Q = 0.1287. Leave this field
            unassigned or make it negative if you don’t want the correction
            done.

        horifoc : bool
            A flag that is set to 1 if a horizontally focusing analyzer is used
            (Section II D). In this case ``hcol[2]`` (see below) is the angular
            size of the analyzer, as seen from the sample position. If the
            field is unassigned or equal to -1, a flat analyzer is assumed.
            Note that this option is only available with the Cooper-Nathans
            method.

        dir : int
            Direction of the crystal (left or right, -1 or +1, respectively).
            Default: -1 (left-handed coordinate frame).

        rh : float
            Horizontal curvature of the analyzer in cm.

        rv : float
            Vertical curvature of the analyzer in cm.

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
    def moncor(self):
        '''Selects the type of normalization used to calculate ``R0``
        If ``moncor=1`` or left undefined, ``R0`` is calculated in
        normalization to monitor counts (Section II C 2). 1/k\ :sub:`i` monitor
        efficiency correction is included automatically. To normalize ``R0`` to
        source flux (Section II C 1), use ``moncor=0``.
        '''
        return self._moncar

    @moncor.setter
    def moncor(self, value):
        self._moncar = value

    @property
    def hcol(self):
        r''' The horizontal Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. In case of a
        horizontally-focusing analyzer ``hcol[2]`` is the angular size of the
        analyzer, as seen from the sample position. If the beam divergence is
        limited by a neutron guide, the corresponding element of :attr:`hcol`
        is the negative of the guide’s *m*-value. For example, for a 58-Ni
        guide ( *m* = 1.2 ) before the monochromator, ``hcol[0]`` should be
        -1.2.
        '''
        return self._hcol

    @hcol.setter
    def hcol(self, value):
        self._hcol = value

    @property
    def vcol(self):
        '''The vertical Soller collimations in minutes of arc (FWHM beam
        divergence) starting from the in-pile collimator. If the beam
        divergence is limited by a neutron guide, the corresponding element of
        :attr:`vcol` is the negative of the guide’s *m*-value. For example, for
        a 58-Ni guide ( *m* = 1.2 ) before the monochromator, ``vcol[0]``
        should be -1.2.
        '''
        return self._vcol

    @vcol.setter
    def vcol(self, value):
        self._vcol = value

    @property
    def arms(self):
        '''distances between the source and monochromator, monochromator
        and sample, sample and analyzer, analyzer and detector, and
        monochromator and monitor, respectively. The 5th element is only needed
        if ``moncor=1``
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

        Parameters
        ----------
        mosaic
            FWHM sample mosaic in the scattering plane
            in minutes of arc. If left unassigned, no sample
            mosaic corrections (section II E) are performed.

        vmosaic
            The vertical sample mosaic in minutes of arc.
            If left unassigned, isotropic mosaic is assumed.

        dir
            The direction of the crystal (left or right, -1 or +1,
            respectively). Default: -1 (left-handed coordinate frame).

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
        return self._sample.u

    @orient1.setter
    def orient1(self, value):
        self._sample.u = np.array(value)

    @property
    def orient2(self):
        '''Miller indexes of the second reciprocal-space orienting vector
        for the S coordinate system, as explained in Section II G.
        '''
        return self._sample.v

    @orient2.setter
    def orient2(self, value):
        self._sample.v = np.array(value)

    @property
    def infin(self):
        '''a flag set to -1 or left unassigned if the final energy is fixed, or
        set to +1 in a fixed-incident setup.
        '''
        return self._infin

    @infin.setter
    def infin(self, value):
        self._infin = value

    @property
    def guide(self):
        r'''A structure that describes the source
        '''
        return self._guide

    @guide.setter
    def guide(self, value):
        self._guide = value

    @property
    def detector(self):
        '''A structure that describes the detector
        '''
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def monitor(self):
        '''A structure that describes the monitor
        '''
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def Smooth(self):
        u'''Defines the smoothing parameters as explained in Section II H. Leave this
        field unassigned if you don’t want this correction done.

        * ``Smooth.E`` is the smoothing FWHM in energy (meV). A small number
          means “no smoothing along this direction”.

        * ``Smooth.X`` is the smoothing FWHM along the first orienting vector
          (x0 axis) in Å\ :sup:`-1`.

        * ``Smooth.Y`` is the smoothing FWHM along the y axis in Å\ :sup:`-1`.

        * ``Smooth.Z`` is the smoothing FWHM along the vertical direction in
          Å\ :sup:`-1`.

        '''
        return self._Smooth

    @Smooth.setter
    def Smooth(self, value):
        self._Smooth = value

    def get_lattice(self):
        r'''Extracts lattice parameters from EXP and returns the direct and
        reciprocal lattice parameters in the form used by _scalar.m, _star.m,
        etc.

        Returns
        -------
        [lattice, rlattice] : [class, class]
            Returns the direct and reciprocal lattice sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        '''
        lattice = Sample(self.sample.a,
                         self.sample.b,
                         self.sample.c,
                         self.sample.alpha * np.pi / 180,
                         self.sample.beta * np.pi / 180,
                         self.sample.gamma * np.pi / 180)
        V, Vstar, rlattice = _star(lattice)  # @UnusedVariable

        return [lattice, rlattice]

    def _StandardSystem(self):
        r'''Returns rotation matrices to calculate resolution in the sample view
        instead of the instrument view

        Parameters
        ----------
        EXP : class
            Instrument class

        Returns
        -------
        [x, y, z, lattice, rlattice] : [array, array, array, class, class]
            Returns the rotation matrices and real and reciprocal lattice
            sample classes

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        '''
        [lattice, rlattice] = self.get_lattice()

        orient1 = self.orient1
        orient2 = self.orient2

        modx = _modvec([orient1[0], orient1[1], orient1[2]], rlattice)

        x = orient1 / modx

        proj = _scalar([orient2[0], orient2[1], orient2[2]], [x[0], x[1], x[2]], rlattice)

        y = orient2 - x * proj

        mody = _modvec([y[0], y[1], y[2]], rlattice)

        if len(np.where(mody <= 0)[0]) > 0:
            raise ValueError('??? Fatal error: Orienting vectors are colinear!')

        y /= mody

        z = np.zeros(3, dtype=np.float64)
        z[0] = x[1] * y[2] - y[1] * x[2]
        z[1] = x[2] * y[0] - y[2] * x[0]
        z[2] = -x[1] * y[0] + y[1] * x[0]

        proj = _scalar([z[0], z[1], z[2]], [x[0], x[1], x[2]], rlattice)

        z = z - x * proj

        proj = _scalar([z[0], z[1], z[2]], [y[0], y[1], y[2]], rlattice)

        z = z - y * proj

        modz = _modvec([z[0], z[1], z[2]], rlattice)

        z /= modz

        return [x, y, z, lattice, rlattice]

    def calc_resolution_in_Q_coords(self, Q, W):
        r'''For a momentum transfer Q and energy transfers W, given experimental
        conditions specified in EXP, calculates the Cooper-Nathans or Popovici
        resolution matrix RM and resolution prefactor R0 in the Q coordinate
        system (defined by the scattering vector and the scattering plane).

        Parameters
        ----------
        Q : ndarray or list of ndarray
            The Q vectors in reciprocal space at which resolution should be
            calculated, in inverse angstroms

        W : float or list of floats
            The energy transfers at which resolution should be calculated in meV

        Returns
        -------
        [R0, RM] : list(float, ndarray)
            Resolution pre-factor (R0) and resolution matrix (RM) at the given
            reciprocal lattice vectors and energy transfers

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        '''
        CONVERT1 = np.pi / 60. / 180. / np.sqrt(8 * np.log(2))
        CONVERT2 = 2.072

        [length, Q, W] = _CleanArgs(Q, W)

        RM = np.zeros((4, 4, length), dtype=np.float64)
        R0 = np.zeros(length, dtype=np.float64)
        RM_ = np.zeros((4, 4), dtype=np.float64)
        D = np.matrix(np.zeros((8, 13), dtype=np.float64))
        d = np.matrix(np.zeros((4, 7), dtype=np.float64))
        T = np.matrix(np.zeros((4, 13), dtype=np.float64))
        t = np.matrix(np.zeros((2, 7), dtype=np.float64))
        A = np.matrix(np.zeros((6, 8), dtype=np.float64))
        C = np.matrix(np.zeros((4, 8), dtype=np.float64))
        Bmatrix = np.matrix(np.zeros((4, 6), dtype=np.float64))

        # the method to use
        method = 0
        if hasattr(self, 'method'):
            method = self.method

        # Assign default values and decode parameters
        moncor = 1
        if hasattr(self, 'moncor'):
            moncor = self.moncor

        alpha = np.array(self.hcol) * CONVERT1
        beta = np.array(self.vcol) * CONVERT1
        mono = self.mono
        etam = np.array(mono.mosaic) * CONVERT1
        etamv = np.copy(etam)
        if hasattr(mono, 'vmosaic') and (method == 1 or method == 'Popovici'):
            etamv = np.array(mono.vmosaic) * CONVERT1

        ana = self.ana
        etaa = np.array(ana.mosaic) * CONVERT1
        etaav = np.copy(etaa)
        if hasattr(ana, 'vmosaic'):
            etaav = np.array(ana.vmosaic) * CONVERT1

        sample = self.sample
        infin = -1
        if hasattr(self, 'infin'):
            infin = self.infin

        efixed = self.efixed

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
        sshapes = np.repeat(np.eye(3, dtype=np.float64)[np.newaxis].reshape((1, 3, 3)), length, axis=0)
        sshape_factor = 12.
        L0 = 1.
        L1 = 1.
        L1mon = 1.
        L2 = 1.
        L3 = 1.
        monorv = 1.e6
        monorh = 1.e6
        anarv = 1.e6
        anarh = 1.e6

        if hasattr(self, 'guide'):
            beam = self.guide
            if hasattr(beam, 'width'):
                beamw = beam.width ** 2 / 12.

            if hasattr(beam, 'height'):
                beamh = beam.height ** 2 / 12.

        bshape = np.diag([beamw, beamh])
        if hasattr(self, 'monitor'):
            monitor = self.monitor
            if hasattr(monitor, 'width'):
                monitorw = monitor.width ** 2 / 12.

            monitorh = monitorw
            if hasattr(monitor, 'height'):
                monitorh = monitor.height ** 2 / 12.

        monitorshape = np.diag([monitorw, monitorh])
        if hasattr(self, 'detector'):
            detector = self.detector
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
        if hasattr(sample, 'shape_type'):
            if sample.shape_type == 'cylindrical':
                sshape_factor = 16.
            elif sample.shape_type == 'rectangular':
                sshape_factor = 12.
        if hasattr(sample, 'width') and hasattr(sample, 'depth') and hasattr(sample, 'height'):
            _sshape = np.diag([sample.depth, sample.width, sample.height]).astype(np.float64) ** 2 / sshape_factor
            sshapes = np.repeat(_sshape[np.newaxis].reshape((1, 3, 3)), length, axis=0)
        elif hasattr(sample, 'shape'):
            _sshape = sample.shape.astype(np.float64) / sshape_factor
            if len(_sshape.shape) == 2:
                sshapes = np.repeat(_sshape[np.newaxis].reshape((1, 3, 3)), length, axis=0)
            else:
                sshapes = _sshape

        if hasattr(self, 'arms') and method == 1:
            arms = self.arms
            L0 = arms[0]
            L1 = arms[1]
            L2 = arms[2]
            L3 = arms[3]
            L1mon = np.copy(L1)
            if len(arms) > 4:
                L1mon = np.copy(arms[4])

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
        if hasattr(self, 'horifoc'):
            horifoc = self.horifoc

        if horifoc == 1:
            alpha[2] = alpha[2] * np.sqrt(8. * np.log(2.) / 12.)

        sm = self.mono.dir
        ss = self.sample.dir
        sa = self.ana.dir

        for ind in range(length):
            sshape = sshapes[ind, :, :]
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

            thetam = np.arcsin(taum / (2. * ki)) * sm
            thetaa = np.arcsin(taua / (2. * kf)) * sa
            s2theta = np.arccos(np.complex((ki ** 2 + kf ** 2 - q ** 2) / (2. * ki * kf))) * ss
            if np.abs(np.imag(s2theta)) > 1e-12:
                raise ValueError(': KI,KF,Q triangle will not close (kinematic equations). Change the value of KFIX,FX,QH,QK or QL.')
            else:
                s2theta = np.real(s2theta)

            # correct sign of curvatures
            monorh = monorh * sm
            monorv = monorv * sm
            anarh = anarh * sa
            anarv = anarv * sa

            thetas = s2theta / 2.
            phi = np.arctan2(-kf * np.sin(s2theta), ki - kf * np.cos(s2theta))

            # Calculate beam divergences defined by neutron guides
            for i, ang in enumerate(alpha):
                if ang < 0:
                    alpha[i] = -alpha[i] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)

            for i, ang, in enumerate(beta):
                if ang < 0:
                    beta[i] = -beta[i] * 0.1 * 60. * (2. * np.pi / ki) / 0.427 / np.sqrt(3.)

            # Redefine sample geometry
            psi = thetas - phi  # Angle from sample geometry X axis to Q
            rot = np.matrix(np.zeros((3, 3), dtype=np.float64))
            rot[0, 0] = np.cos(psi)
            rot[1, 1] = np.cos(psi)
            rot[0, 1] = np.sin(psi)
            rot[1, 0] = -np.sin(psi)
            rot[2, 2] = 1.

            # sshape=rot'*sshape*rot
            sshape = np.matrix(rot) * np.matrix(sshape) * np.matrix(rot).H

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

            # Definition of matrix Bmatrix
            Bmatrix[0, 0] = np.cos(phi)
            Bmatrix[0, 1] = np.sin(phi)
            Bmatrix[0, 3] = -np.cos(phi - s2theta)
            Bmatrix[0, 4] = -np.sin(phi - s2theta)
            Bmatrix[1, 0] = -Bmatrix[0, 1]
            Bmatrix[1, 1] = Bmatrix[0, 0]
            Bmatrix[1, 3] = -Bmatrix[0, 4]
            Bmatrix[1, 4] = Bmatrix[0, 3]
            Bmatrix[2, 2] = 1.
            Bmatrix[2, 5] = -1.
            Bmatrix[3, 0] = 2. * CONVERT2 * ki
            Bmatrix[3, 3] = -2. * CONVERT2 * kf

            # Definition of matrix S
            Sinv = np.matrix(blkdiag(np.array(bshape, dtype=np.float64), mshape, sshape, ashape, dshape))  # S-1 matrix
            S = Sinv.I

            # Definition of matrix T
            T[0, 0] = -1. / (2. * L0)  # mistake in paper
            T[0, 2] = np.cos(thetam) * (1. / L1 - 1. / L0) / 2.
            T[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1 - 2. / (monorh * np.sin(thetam))) / 2.
            T[0, 5] = np.sin(thetas) / (2. * L1)
            T[0, 6] = np.cos(thetas) / (2. * L1)
            T[1, 1] = -1. / (2. * L0 * np.sin(thetam))
            T[1, 4] = (1. / L0 + 1. / L1 - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
            T[1, 7] = -1. / (2. * L1 * np.sin(thetam))
            T[2, 5] = np.sin(thetas) / (2. * L2)
            T[2, 6] = -np.cos(thetas) / (2. * L2)
            T[2, 8] = np.cos(thetaa) * (1. / L3 - 1. / L2) / 2.
            T[2, 9] = np.sin(thetaa) * (1. / L2 + 1. / L3 - 2. / (anarh * np.sin(thetaa))) / 2.
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
            if method == 1 or method == 'popovici':
                K = S + T.H * F * T
                H = np.linalg.inv(D * np.linalg.inv(K) * D.H)
                Ninv = A * np.linalg.inv(H + G) * A.H
            else:
                H = G + C.H * F * C
                Ninv = A * np.linalg.inv(H) * A.H
                # Horizontally focusing analyzer if needed
                if horifoc > 0:
                    Ninv = np.linalg.inv(Ninv)
                    Ninv[4, 4] = (1 / (kf * alpha[2])) ** 2
                    Ninv[4, 3] = 0
                    Ninv[3, 4] = 0
                    Ninv[3, 3] = (np.tan(thetaa) / (etaa * kf)) ** 2
                    Ninv = np.linalg.inv(Ninv)

            Minv = Bmatrix * Ninv * Bmatrix.H

            M = np.linalg.inv(Minv)
            RM_ = np.copy(M)

            # Calculation of prefactor, normalized to source
            Rm = ki ** 3 / np.tan(thetam)
            Ra = kf ** 3 / np.tan(thetaa)
            R0_ = Rm * Ra * (2. * np.pi) ** 4 / (64. * np.pi ** 2 * np.sin(thetam) * np.sin(thetaa))

            if method == 1 or method == 'popovici':
                # Popovici
                R0_ = R0_ * np.sqrt(np.linalg.det(F) / np.linalg.det(H + G))
            else:
                # Cooper-Nathans (popovici Eq 5 and 9)
                R0_ = R0_ * np.sqrt(np.linalg.det(F) / np.linalg.det(H))

            # Normalization to flux on monitor
            if moncor == 1:
                g = G[0:4, 0:4]
                f = F[0:2, 0:2]
                c = C[0:2, 0:4]
                t[0, 0] = -1. / (2. * L0)  # mistake in paper
                t[0, 2] = np.cos(thetam) * (1. / L1mon - 1. / L0) / 2.
                t[0, 3] = np.sin(thetam) * (1. / L0 + 1. / L1mon - 2. / (monorh * np.sin(thetam))) / 2.
                t[0, 6] = 1. / (2. * L1mon)
                t[1, 1] = -1. / (2. * L0 * np.sin(thetam))
                t[1, 4] = (1. / L0 + 1. / L1mon - 2. * np.sin(thetam) / monorv) / (2. * np.sin(thetam))
                sinv = blkdiag(np.array(bshape, dtype=np.float64), mshape, monitorshape)  # S-1 matrix
                s = np.linalg.inv(sinv)
                d[0, 0] = -1. / L0
                d[0, 2] = -np.cos(thetam) / L0
                d[0, 3] = np.sin(thetam) / L0
                d[2, 1] = D[0, 0]
                d[2, 4] = -D[0, 0]
                d[1, 2] = np.cos(thetam) / L1mon
                d[1, 3] = np.sin(thetam) / L1mon
                d[1, 5] = 0.
                d[1, 6] = 1. / L1mon
                d[3, 4] = -1. / L1mon
                if method == 1 or method == 'popovici':
                    # Popovici
                    Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * np.sqrt(np.linalg.det(f) / np.linalg.det(np.linalg.inv(d * np.linalg.inv(s + t.H * f * t) * d.H) + g))
                else:
                    # Cooper-Nathans
                    Rmon = Rm * (2 * np.pi) ** 2 / (8 * np.pi * np.sin(thetam)) * np.sqrt(np.linalg.det(f) / np.linalg.det(g + c.H * f * c))

                R0_ = R0_ / Rmon
                R0_ = R0_ * ki  # 1/ki monitor efficiency

            # Transform prefactor to Chesser-Axe normalization
            R0_ = R0_ / (2. * np.pi) ** 2 * np.sqrt(np.linalg.det(RM_))
            # Include kf/ki part of cross section
            R0_ = R0_ * kf / ki

            # Take care of sample mosaic if needed
            # [S. A. Werner & R. Pynn, J. Appl. Phys. 42, 4736, (1971), eq 19]
            if hasattr(sample, 'mosaic'):
                etas = sample.mosaic * CONVERT1
                etasv = np.copy(etas)
                if hasattr(sample, 'vmosaic'):
                    etasv = sample.vmosaic * CONVERT1
                R0_ = R0_ / np.sqrt((1 + (q * etas) ** 2 * RM_[2, 2]) * (1 + (q * etasv) ** 2 * RM_[1, 1]))
                Minv[1, 1] = Minv[1, 1] + q ** 2 * etas ** 2
                Minv[2, 2] = Minv[2, 2] + q ** 2 * etasv ** 2
                RM_ = np.linalg.inv(Minv)

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
            RM[:, :, ind] = np.copy(RM_[:, :])

        return [R0, RM]

    def calc_resolution(self, hkle):
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
            Translated from ResLib, originally authored by A. Zheludev, 1999-2007,
            Oak Ridge National Laboratory

        '''
        self.HKLE = hkle
        [H, K, L, W] = hkle

        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        self.H, self.K, self.L, self.W = H, K, L, W

        [x, y, z, sample, rsample] = self._StandardSystem()

        Q = _modvec([H, K, L], rsample)
        uq0 = H / Q  # Unit vector along Q
        uq1 = K / Q
        uq2 = L / Q
        uq = np.vstack((uq0, uq1, uq2))

        xq = _scalar([x[0], x[1], x[2]], [uq[0], uq[1], uq[2]], rsample)
        yq = _scalar([y[0], y[1], y[2]], [uq[0], uq[1], uq[2]], rsample)
        zq = 0  # scattering vector assumed to be in (orient1,orient2) plane

        tmat = np.zeros((4, 4, length), dtype=np.float64)  # Coordinate transformation matrix
        tmat[3, 3, :] = 1.
        tmat[2, 2, :] = 1.
        tmat[0, 0, :] = xq
        tmat[0, 1, :] = yq
        tmat[1, 1, :] = xq
        tmat[1, 0, :] = -yq

        RMS = np.zeros((4, 4, length), dtype=np.float64)
        rot = np.zeros((3, 3), dtype=np.float64)

        # Sample shape matrix in coordinate system defined by scattering vector
        sample = self.sample
        if hasattr(sample, 'shape'):
            samples = []
            for i in range(length):
                rot[0, 0] = tmat[0, 0, i]
                rot[1, 0] = tmat[1, 0, i]
                rot[0, 1] = tmat[0, 1, i]
                rot[1, 1] = tmat[1, 1, i]
                rot[2, 2] = tmat[2, 2, i]
                samples.append(np.matrix(rot) * np.matrix(sample.shape) * np.matrix(rot).H)
            self.sample.shape = np.array(samples)

        [R0, RM] = self.calc_resolution_in_Q_coords(Q, W)

        for i in range(length):
            RMS[:, :, i] = np.matrix(tmat[:, :, i]).H * np.matrix(RM[:, :, i]) * np.matrix(tmat[:, :, i])

        mul = np.zeros((4, 4))
        e = np.identity(4)
        for i in range(length):
            if hasattr(self, 'Smooth'):
                if self.Smooth.X:
                    mul[0, 0] = 1 / (self.Smooth.X ** 2 / 8 / np.log(2))
                    mul[1, 1] = 1 / (self.Smooth.Y ** 2 / 8 / np.log(2))
                    mul[2, 2] = 1 / (self.Smooth.E ** 2 / 8 / np.log(2))
                    mul[3, 3] = 1 / (self.Smooth.Z ** 2 / 8 / np.log(2))
                    R0[i] = R0[i] / np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(RMS[:, :, i]))) * np.sqrt(np.linalg.det(np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[:, :, i])))
                    RMS[:, :, i] = np.matrix(e) / (np.matrix(e) / np.matrix(mul) + np.matrix(e) / np.matrix(RMS[:, :, i]))

        self.R0, self.RMS, self.RM = [np.squeeze(item) for item in (R0, RMS, RM)]

    def calc_projections(self, hkle, npts=36):
        r'''Calculates the resolution ellipses for projections and slices from
        the resolution matrix.

        Parameters
        ----------
        hkle : list
            Positions at which projections should be calculated.

        npts : int, optional
            Number of points in the outputted ellipse curve

        Returns
        -------
        projections : dictionary
            A dictionary containing projections in the planes: QxQy, QxW, and
            QyW, both projections and slices

        '''
        [H, K, L, W] = hkle
        try:
            if H == self.H and K == self.K and L == self.L and W == self.W:
                NP = np.array(self.RMS)
                R0 = self.R0
            else:
                self.calc_resolution(hkle)
                NP = np.array(self.RMS)
                R0 = self.R0
        except AttributeError:
            self.calc_resolution(hkle)
            NP = np.array(self.RMS)
            R0 = self.R0

        const = 1.17741  # half width factor

        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        hkle = [H, K, L, W]

        self.projections = {'QxQy': np.zeros((2, npts, NP.shape[-1])),
                            'QxQySlice': np.zeros((2, npts, NP.shape[-1])),
                            'QxW': np.zeros((2, npts, NP.shape[-1])),
                            'QxWSlice': np.zeros((2, npts, NP.shape[-1])),
                            'QyW': np.zeros((2, npts, NP.shape[-1])),
                            'QyWSlice': np.zeros((2, npts, NP.shape[-1])),
                            'QxQy_fwhm': np.zeros((2, NP.shape[-1])),
                            'QxQySlice_fwhm': np.zeros((2, NP.shape[-1])),
                            'QxW_fwhm': np.zeros((2, NP.shape[-1])),
                            'QxWSlice_fwhm': np.zeros((2, NP.shape[-1])),
                            'QyW_fwhm': np.zeros((2, NP.shape[-1])),
                            'QyWSlice_fwhm': np.zeros((2, NP.shape[-1]))}

        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()

        o1 = np.copy(self.orient1)
        o2 = np.copy(self.orient2)
        pr = _scalar([o2[0], o2[1], o2[2]], [yvec[0], yvec[1], yvec[2]], rsample)
        o2[0] = yvec[0] * pr
        o2[1] = yvec[1] * pr
        o2[2] = yvec[2] * pr

        if np.abs(o2[0]) < 1e-5:
            o2[0] = 0
        if np.abs(o2[1]) < 1e-5:
            o2[1] = 0
        if np.abs(o2[2]) < 1e-5:
            o2[2] = 0

        if np.abs(o1[0]) < 1e-5:
            o1[0] = 0
        if np.abs(o1[1]) < 1e-5:
            o1[1] = 0
        if np.abs(o1[2]) < 1e-5:
            o1[2] = 0
        frame = '[Q1,Q2,E]'

        A = np.copy(NP)

        if A.shape == (4, 4):
            A = A.reshape((4, 4, 1))
            R0 = R0[np.newaxis]

        for ind in range(A.shape[-1]):
            # Remove the vertical component from the matrix.
            Bmatrix = np.matrix([np.concatenate((A[0, 0:2, ind], [A[0, 3, ind]])), np.concatenate((A[1, 0:2, ind], [A[1, 3, ind]])), np.concatenate((A[3, 0:2, ind], [A[3, 3, ind]]))])

            # Projection into Qx, Qy plane
            [R0P, MP] = project_into_plane(2, R0[ind], Bmatrix)
            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQy_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QxQy_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QxQy'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                          [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                           np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient2 / np.linalg.norm(self.orient2) ** 2)],
                                                          npts=npts)

            # Slice through Qx,Qy plane
            MP = A[:2, :2, ind]

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxQySlice_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QxQySlice_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QxQySlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                               [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                                np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient2 / np.linalg.norm(self.orient2) ** 2)],
                                                               npts=npts)

            # Projection into Qx, W plane

            [R0P, MP] = project_into_plane(1, R0, Bmatrix)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxW_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QxW_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QxW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                         [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                          hkle[3][ind]],
                                                         npts=npts)

            # Slice through Qx,W plane
            MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QxWSlice_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QxWSlice_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QxWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                              [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient1 / np.linalg.norm(self.orient1) ** 2),
                                                               hkle[3][ind]],
                                                              npts=npts)

            # Projections into Qy, W plane
            [R0P, MP] = project_into_plane(0, R0, Bmatrix)

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyW_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QyW_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QyW'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                         [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient2 / np.linalg.norm(self.orient2) ** 2),
                                                          hkle[3][ind]],
                                                         npts=npts)

            # Slice through Qy,W plane
            MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])

            theta = 0.5 * np.arctan2(2 * MP[0, 1], (MP[0, 0] - MP[1, 1]))
            S = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            MP = np.matrix(S) * np.matrix(MP) * np.matrix(S).H

            hwhm_xp = const / np.sqrt(MP[0, 0])
            hwhm_yp = const / np.sqrt(MP[1, 1])

            self.projections['QyWSlice_fwhm'][0, ind] = 2 * hwhm_xp
            self.projections['QyWSlice_fwhm'][1, ind] = 2 * hwhm_yp

            self.projections['QyWSlice'][:, :, ind] = ellipse(hwhm_xp, hwhm_yp, theta,
                                                              [np.dot([hkle[0][ind], hkle[1][ind], hkle[2][ind]], self.orient2 / np.linalg.norm(self.orient2) ** 2),
                                                               hkle[3][ind]],
                                                              npts=npts)

    def get_angles_and_Q(self, hkle):
        r'''Returns the Triple Axis Spectrometer angles and Q-vector given
        position in reciprocal space

        Parameters
        ----------
        hkle : list
            Array of the scattering vector and energy transfer at which the
            calculation should be performed

        Returns
        -------
        [A, Q] : list
            The angles A (A1 -- A5 in a list of floats) and Q (ndarray)

        '''
        # compute all TAS angles (in plane)

        h, k, l, w = hkle
        # compute angles
        try:
            fx = 2 * int(self.infin == -1) + int(self.infin == 1)
        except AttributeError:
            fx = 2

        kfix = Energy(energy=self.efixed).wavevector
        f = 0.4826  # f converts from energy units into k^2, f=0.4826 for meV
        ki = np.sqrt(kfix ** 2 + (fx - 1) * f * w)  # kinematical equations.
        kf = np.sqrt(kfix ** 2 - (2 - fx) * f * w)

        # compute the transversal Q component, and A3 (sample rotation)
        # from McStas templateTAS.instr and TAS MAD ILL
        a = np.array([self.sample.a, self.sample.b, self.sample.c]) / (2 * np.pi)
        alpha = np.deg2rad([self.sample.alpha, self.sample.beta, self.sample.gamma])
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cc = np.sum(cosa * cosa)
        cc = 1 + 2 * np.product(cosa) - cc
        cc = np.sqrt(cc)
        b = sina / (a * cc)
        c1 = np.roll(cosa[np.newaxis].T, -1)
        c2 = np.roll(c1, -1)
        s1 = np.roll(sina[np.newaxis].T, -1)
        s2 = np.roll(s1, -1)
        cosb = (c1 * c2 - cosa[np.newaxis].T) / (s1 * s2)
        sinb = np.sqrt(1 - cosb * cosb)

        bb = np.array([[b[0], 0, 0],
                       [b[1] * cosb[2], b[1] * sinb[2], 0],
                       [b[2] * cosb[1], -b[2] * sinb[1] * cosa[0], 1 / a[2]]])
        bb = bb.T

        aspv = np.hstack((self.orient1[np.newaxis].T, self.orient2[np.newaxis].T))

        vv = np.zeros((3, 3))
        vv[0:2, :] = np.transpose(np.dot(bb, aspv))
        for m in range(2, 0, -1):
            vt = np.roll(np.roll(vv, -1, axis=0), -1, axis=1) * np.roll(np.roll(vv, -2, axis=0), -2, axis=1) - np.roll(np.roll(vv, -1, axis=0), -2, axis=1) * np.roll(np.roll(vv, -2, axis=0), -1, axis=1)
            vv[m, :] = vt[m, :]

        c = np.sqrt(np.sum(vv * vv, axis=0))

        vv = vv / np.tile(c, (3, 1))
        s = vv.T * bb

        qt = np.squeeze(np.dot(np.array([h, k, l]).T, s.T))

        qs = np.sum(qt ** 2)
        Q = np.sqrt(qs)

        sm = self.mono.dir
        ss = self.sample.dir
        sa = self.ana.dir
        dm = 2 * np.pi / GetTau(self.mono.tau)
        da = 2 * np.pi / GetTau(self.ana.tau)
        thetaa = sa * np.arcsin(np.pi / (da * kf))  # theta angles for analyser
        thetam = sm * np.arcsin(np.pi / (dm * ki))  # and monochromator.
        thetas = ss * 0.5 * np.arccos((ki ** 2 + kf ** 2 - Q ** 2) / (2 * ki * kf))  # scattering angle from sample.

        A3 = -np.arctan2(qt[1], qt[0]) - np.arccos((np.dot(kf, kf) - np.dot(Q, Q) - np.dot(ki, ki)) / (-2 * np.dot(Q, ki)))
        A3 = ss * A3

        A1 = thetam
        A2 = 2 * A1
        A4 = 2 * thetas
        A5 = thetaa
        A6 = 2 * A5

        A = np.squeeze(np.rad2deg([A1, A2, A3, A4, A5, A6]))

        return [A, Q]

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

        try:
            A = self.RMS
        except:
            self.calc_resolution(hkle)
            A = self.RMS

        ind = np.where((self.H == hkle[0]) & (self.K == hkle[1]) & (self.L == hkle[2]) & (self.W == hkle[3]))
        if len(ind[0]) == 0:
            raise ValueError('Resolution at provided HKLE has not been calculated.')

        ind = ind[0][0]

        if len(A.shape) != 3:
            A = A.reshape((A.shape[0], A.shape[1], 1))
            selfR0 = self.R0[np.newaxis]

        # Remove the vertical component from the matrix
        Bmatrix = np.vstack((np.hstack((A[0, :2:1, ind], A[0, 3, ind])),
                             np.hstack((A[1, :2:1, ind], A[1, 3, ind])),
                             np.hstack((A[3, :2:1, ind], A[3, 3, ind]))))

        if plane == 'QxQy':
            R0 = np.sqrt(2 * np.pi / Bmatrix[2, 2]) * selfR0[ind]
            if mode == 'project':
                # Projection into Qx, Qy plane
                R0, MP = project_into_plane(2, R0, Bmatrix)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,Qy plane
                MP = np.array(A[:2:1, :2:1, ind])
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QxW':
            R0 = np.sqrt(2 * np.pi / Bmatrix[1, 1]) * selfR0[ind]
            if mode == 'project':
                # Projection into Qx, W plane
                R0, MP = project_into_plane(1, R0, Bmatrix)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qx,W plane
                MP = np.array([[A[0, 0, ind], A[0, 3, ind]], [A[3, 0, ind], A[3, 3, ind]]])
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])

        if plane == 'QyW':
            R0 = np.sqrt(2 * np.pi / Bmatrix[0, 0]) * selfR0[ind]
            if mode == 'project':
                # Projections into Qy, W plane
                R0, MP = project_into_plane(0, R0, Bmatrix)
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])
            if mode == 'slice':
                # Slice through Qy,W plane
                MP = np.array([[A[1, 1, ind], A[1, 3, ind]], [A[3, 1, ind], A[3, 3, ind]]])
                return (R0, MP[0, 0], MP[1, 1], MP[0, 1])

    def resolution_convolution(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r'''Numerically calculate the convolution of a user-defined
        cross-section function with the resolution function for a
        3-axis neutron scattering experiment.

        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 4D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, $\phi_2$, and $\phi_3$, and 2*ACCURACY[1]+1
            along $\phi_4$ (vertical direction). 'mc': 4D Monte Carlo
            integration. The cross section is sampled in 1000*ACCURACY
            randomly chosen points, uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        '''
        self.calc_resolution(hkle)
        [R0, RMS] = [np.copy(self.R0), np.copy(self.RMS)]

        H, K, L, W = hkle
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)
        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()

        Mzz = RMS[2, 2, :]
        Mww = RMS[3, 3, :]
        Mxx = RMS[0, 0, :]
        Mxy = RMS[0, 1, :]
        Mxw = RMS[0, 3, :]
        Myy = RMS[1, 1, :]
        Myw = RMS[1, 3, :]

        Mxx = Mxx - Mxw ** 2. / Mww
        Mxy = Mxy - Mxw * Myw / Mww
        Myy = Myy - Myw ** 2. / Mww
        MMxx = Mxx - Mxy ** 2. / Myy

        detM = MMxx * Myy * Mzz * Mww

        tqz = 1. / np.sqrt(Mzz)
        tqx = 1. / np.sqrt(MMxx)
        tqyy = 1. / np.sqrt(Myy)
        tqyx = -Mxy / Myy / np.sqrt(MMxx)
        tqww = 1. / np.sqrt(Mww)
        tqwy = -Myw / Mww / np.sqrt(Myy)
        tqwx = -(Mxw / Mww - Myw / Mww * Mxy / Myy) / np.sqrt(MMxx)

        inte = sqw(H, K, L, W, p)
        [modes, points] = inte.shape

        if pref is None:
            prefactor = np.ones((modes, points))
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Invalid number or output arguments in prefactor function, pref')

        found = 0
        if METHOD == 'fix':
            found = 1
            if ACCURACY is None:
                ACCURACY = np.array([7, 0])
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cw, cx, cy] = np.meshgrid(dd1, dd1, dd1, indexing='ij')
            tx = np.tan(cx)
            ty = np.tan(cy)
            tw = np.tan(cw)
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * np.exp(-0.5 * (tw ** 2)) * (1 + tw ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)

            for iz in range(len(tz)):
                for i in range(length):
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm * normz[iz]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 3 * step2 / np.sqrt(detM)
            if M[1] == 0:
                conv = conv * 0.79788
            if M[0] == 0:
                conv = conv * 0.79788 ** 3

        if METHOD == 'mc':
            found = 1
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError('ACCURACY must be an int when using Monte Carlo method')
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r = np.random.randn(4, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    cw = r[3, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    tw = np.tan(cw)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2 + tw ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2) * (1 + tw ** 2)
                    dQ1 = tqx[i] * tx
                    dQ2 = tqyy[i] * ty + tqyx[i] * tx
                    dW = tqwx[i] * tx + tqwy[i] * ty + tqww[i] * tw
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    W1 = W[i] + dW
                    inte = sqw(H1, K1, L1, W1, p)
                    for j in range(modes):
                        add = inte[j, :] * norm
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000 * np.pi ** 4. / np.sqrt(detM)

        if found == 0:
            raise ValueError('Unknown convolution METHOD. Valid options are: "fix",  "mc".')

        conv = conv * R0
        conv = conv + bgr

        return conv

    def resolution_convolution_SMA(self, sqw, pref, nargout, hkle, METHOD='fix', ACCURACY=None, p=None, seed=None):
        r'''Numerically calculate the convolution of a user-defined single-mode
        cross-section function with the resolution function for a 3-axis
        neutron scattering experiment.

        Parameters
        ----------
        sqw : func
            User-supplied "fast" model cross section.

        pref : func
            User-supplied "slow" cross section prefactor and background
            function.

        nargout : int
            Number of arguments returned by the pref function

        hkle : tup
            Tuple of H, K, L, and W, specifying the wave vector and energy
            transfers at which the convolution is to be calculated (i.e.
            define $\mathbf{Q}_0$). H, K, and L are given in reciprocal
            lattice units and W in meV.

        EXP : obj
            Instrument object containing all information on experimental setup.

        METHOD : str
            Specifies which 3D-integration method to use. 'fix' (Default):
            sample the cross section on a fixed grid of points uniformly
            distributed $\phi$-space. 2*ACCURACY[0]+1 points are sampled
            along $\phi_1$, and $\phi_2$, and 2*ACCURACY[1]+1 along $\phi_3$
            (vertical direction). 'mc': 3D Monte Carlo integration. The cross
            section is sampled in 1000*ACCURACY randomly chosen points,
            uniformly distributed in $\phi$-space.

        ACCURACY : array(2) or int
            Determines the number of sampling points in the integration.

        p : list
            A parameter that is passed on, without change to sqw and pref.

        Returns
        -------
        conv : array
            Calculated value of the cross section, folded with the resolution
            function at the given $\mathbf{Q}_0$

        Notes
        -----
        Translated from ResLib 3.4c, originally authored by A. Zheludev,
        1999-2007, Oak Ridge National Laboratory

        '''
        self.calc_resolution(hkle)
        [R0, RMS] = [np.copy(self.R0), np.copy(self.RMS)]

        H, K, L, W = hkle
        [length, H, K, L, W] = _CleanArgs(H, K, L, W)

        [xvec, yvec, zvec, sample, rsample] = self._StandardSystem()

        Mww = RMS[3, 3, :]
        Mxw = RMS[0, 3, :]
        Myw = RMS[1, 3, :]

        GammaFactor = np.sqrt(Mww / 2)
        OmegaFactorx = Mxw / np.sqrt(2 * Mww)
        OmegaFactory = Myw / np.sqrt(2 * Mww)

        Mzz = RMS[2, 2, :]
        Mxx = RMS[0, 0, :]
        Mxx = Mxx - Mxw ** 2 / Mww
        Myy = RMS[1, 1, :]
        Myy = Myy - Myw ** 2 / Mww
        Mxy = RMS[0, 1, :]
        Mxy = Mxy - Mxw * Myw / Mww

        detxy = np.sqrt(Mxx * Myy - Mxy ** 2)
        detz = np.sqrt(Mzz)

        tqz = 1. / detz
        tqy = np.sqrt(Mxx) / detxy
        tqxx = 1. / np.sqrt(Mxx)
        tqxy = Mxy / np.sqrt(Mxx) / detxy

        [disp, inte, WL0] = sqw(H, K, L, p)
        [modes, points] = disp.shape

        if pref is None:
            prefactor = np.ones(modes, points)
            bgr = 0
        else:
            if nargout == 2:
                [prefactor, bgr] = pref(H, K, L, W, self, p)
            elif nargout == 1:
                prefactor = pref(H, K, L, W, self, p)
                bgr = 0
            else:
                raise ValueError('Fatal error: invalid number or output arguments in prefactor function')

        found = 0
        if METHOD == 'mc':
            found = 1
            if isinstance(ACCURACY, (list, np.ndarray, tuple)):
                if len(ACCURACY) == 1:
                    ACCURACY = ACCURACY[0]
                else:
                    raise ValueError('ACCURACY must be an int when using Monte Carlo method')
            if ACCURACY is None:
                ACCURACY = 10
            M = ACCURACY
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            for i in range(length):
                for MonteCarlo in range(M):
                    if seed is not None:
                        np.random.seed(seed)
                    r = np.random.randn(3, 1000) * np.pi - np.pi / 2
                    cx = r[0, :]
                    cy = r[1, :]
                    cz = r[2, :]
                    tx = np.tan(cx)
                    ty = np.tan(cy)
                    tz = np.tan(cz)
                    norm = np.exp(-0.5 * (tx ** 2 + ty ** 2 + tz ** 2)) * (1 + tx ** 2) * (1 + ty ** 2) * (1 + tz ** 2)
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv / M / 1000. * np.pi ** 3

        if METHOD == 'fix':
            found = 1
            if ACCURACY is None:
                ACCURACY = [7, 0]
            M = ACCURACY
            step1 = np.pi / (2 * M[0] + 1)
            step2 = np.pi / (2 * M[1] + 1)
            dd1 = np.linspace(-np.pi / 2 + step1 / 2, np.pi / 2 - step1 / 2, (2 * M[0] + 1))
            dd2 = np.linspace(-np.pi / 2 + step2 / 2, np.pi / 2 - step2 / 2, (2 * M[1] + 1))
            convs = np.zeros((modes, length))
            conv = np.zeros(length)
            [cy, cx] = np.meshgrid(dd1, dd1, indexing='ij')
            tx = np.tan(cx.flatten())
            ty = np.tan(cy.flatten())
            tz = np.tan(dd2)
            norm = np.exp(-0.5 * (tx ** 2 + ty ** 2)) * (1 + tx ** 2) * (1 + ty ** 2)
            normz = np.exp(-0.5 * (tz ** 2)) * (1 + tz ** 2)
            for iz in range(tz.size):
                for i in range(length):
                    dQ1 = tqxx[i] * tx - tqxy[i] * ty
                    dQ2 = tqy[i] * ty
                    dQ4 = tqz[i] * tz[iz]
                    H1 = H[i] + dQ1 * xvec[0] + dQ2 * yvec[0] + dQ4 * zvec[0]
                    K1 = K[i] + dQ1 * xvec[1] + dQ2 * yvec[1] + dQ4 * zvec[1]
                    L1 = L[i] + dQ1 * xvec[2] + dQ2 * yvec[2] + dQ4 * zvec[2]
                    [disp, inte, WL] = sqw(H1, K1, L1, p)
                    [modes, points] = disp.shape
                    for j in range(modes):
                        Gamma = WL[j, :] * GammaFactor[i]
                        Omega = GammaFactor[i] * (disp[j, :] - W[i]) + OmegaFactorx[i] * dQ1 + OmegaFactory[i] * dQ2
                        add = inte[j, :] * _voigt(Omega, Gamma) * norm * normz[iz] / detxy[i] / detz[i]
                        convs[j, i] = convs[j, i] + np.sum(add)

                    conv[i] = np.sum(convs[:, i] * prefactor[:, i])

            conv = conv * step1 ** 2 * step2

            if M[1] == 0:
                conv = conv * 0.79788
            if M[0] == 0:
                conv = conv * 0.79788 ** 2

        if found == 0:
            ValueError('??? Error in ConvRes: Unknown convolution method! Valid options are: "fix" or "mc".')

        conv = conv * R0
        conv = conv + bgr

        return conv
