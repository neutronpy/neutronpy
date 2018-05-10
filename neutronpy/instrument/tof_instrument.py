# -*- coding: utf-8 -*-
r"""Define a Time Of Flight instrument for resolution calculations

"""
import numpy as np

from ..constants import e, hbar, neutron_mass
from ..crystal.sample import Sample
from ..energy import Energy
from .chopper import Chopper
from .detector import Detector
from .exceptions import DetectorError
from .guide import Guide
from .plot import PlotTofInstrument
from .tools import chop, get_angle_ki_Q, get_kfree


class TimeOfFlightInstrument(PlotTofInstrument):
    r"""An object representing a Time of Flight (TOF) instrument experimental
    configuration, including a sample.

    Parameters
    ----------
    ei : float, optional
        Incident energy in meV. Default: 3.0

    choppers : obj or list, optional
        Chopper object or list of chopper objects defining all choppers.
        If list, choppers will be automatically sorted by distance from
        source. Default: See code.

    sample : obj, optional
        Sample objection defining the properties of the sample. Default: See code.

    detector : obj, optional
        Detector object defining the properties of the detector. Default: See code.

    guides : obj, optional

    theta_i : float, optional
        Incident 2-theta in degrees. Default: 0

    phi_i : float, optional
        Incident phi in degrees. Default: 0

    Notes
    -----
    If no initial values are specified, the default instrument is similar to
    the T-REX concept for the ESS specified in Violini et al. (2014).

    Calculations of the resolution are based on equations in N. Violini et al.,
    Nuclear Instruments and Methods in Physics Research A 736 (2014) 31-39.

    Attributes
    ----------
    ei
    choppers
    sample
    detector
    guides


    """

    def __init__(self, ei=3.0, choppers=None, sample=None, detector=None, guides=None, theta_i=0, phi_i=0, **kwargs):
        self._ei = Energy(energy=ei)

        if choppers:
            if isinstance(choppers, list):
                self.choppers = sorted(choppers, key=lambda x: x.distance)
            else:
                self.choppers = [choppers]
        else:
            p_chopper = Chopper(0.0, 60.0, 5.0, 10, 'disk', tau=38)
            m_chopper = Chopper(2500.0, 240.0, 2.0, 10, 'disk', 2, tau=6)
            self.choppers = [p_chopper, m_chopper]

        if sample:
            self.sample = sample
        else:
            self.sample = Sample(a=6, b=7, c=8, alpha=90, beta=90, gamma=90, u=[1, 0, 0], v=[0, 1, 0], distance=3600.0)

        if detector:
            self.detector = detector
        else:
            self.detector = Detector('cylindrical', [-90, 90], [-15, 15], 300.0, 0.1, 256)

        if guides:
            self.guides = guides
        else:
            pm_guide = Guide(1, self.choppers[1].distance - self.choppers[0].distance, 15, 15)
            ms_guide = Guide(1, self.sample.distance - self.choppers[1].distance, 15, 15)
            self.guides = [pm_guide, ms_guide]

        self.theta_i = theta_i
        self.phi_i = phi_i

        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

    def __repr__(self):
        return "Instrument('tof', engine='neutronpy', ei={0})".format(self.ei)

    @property
    def ei(self):
        r"""Incident Energy object of type :py:class:`.Energy`

        """
        return self._ei

    @ei.setter
    def ei(self, value):
        self._ei = Energy(energy=value)

    def calc_resolution_in_Q_coords(self, Q, W):
        r"""

        Parameters
        ----------
        Q : list
            Position in Q-coords

        W : float
            Energy transfer

        Returns
        -------
            R0, RM : tuple
                R0 resolution prefactor and RM resolution matrix
        """
        # Definitions
        tau_p = getattr(self, "tau_p", self.choppers[0].tau) / 1e6
        tau_m = getattr(self, "tau_m", self.choppers[1].tau) / 1e6
        tau_d = getattr(self, "tau_d", self.detector.tau) / 1e6

        l_pm = getattr(self, "l_pm", np.abs(np.subtract(self.choppers[0].distance, self.choppers[1].distance))) / 1e2
        l_ms = getattr(self, "l_ms", self.sample.distance) / 1e2
        l_sd = getattr(self, "l_sd", self.detector.radius) / 1e2

        sigma_l_pm = getattr(self, "sigma_l_pm", self.guides[0].sigma_l) / 1e2
        sigma_l_ms = getattr(self, "sigma_l_ms", self.guides[1].sigma_l) / 1e2
        sigma_l_sd = getattr(self, "sigma_l_sd", self.detector.radius / np.cos(np.deg2rad(getattr(self.sample, "mosaic", 60) / 60)) - self.detector.radius) / 1e2

        sigma_theta_i = np.deg2rad(getattr(self, "sigma_theta_i", self.guides[1].sigma_theta * self.ei.wavelength))
        sigma_phi_i = np.deg2rad(getattr(self, "sigma_phi_i", self.guides[1].sigma_phi * self.ei.wavelength))

        sigma_theta = np.deg2rad(getattr(self, "sigma_theta", np.max([getattr(self.sample, "mosaic", 60.0) / 60.0, self.detector.hpixels])))
        sigma_phi = np.deg2rad(getattr(self, "sigma_phi", np.max([getattr(self.sample, "vmosaic", 60.0) / 60.0, np.max(self.detector.height) / self.detector.vpixels])))

        theta_i = np.deg2rad(getattr(self, "theta_i", 0.0))
        phi_i = np.deg2rad(getattr(self, "phi_i", 0.0))

        # Get ki, kf and related from ei and energy transfer
        ki = self.ei.wavevector
        kf = get_kfree(W, ki)

        vi = self.ei.velocity
        vf = Energy(wavevector=kf).velocity

        ti = l_pm / vi
        tf = l_sd / vf

        # Eq. 20 & 21 Violini :: time sigmas
        sigma_t = np.sqrt(tau_p ** 2 + tau_m ** 2)
        sigma_t_md = np.sqrt(tau_m ** 2 + tau_d ** 2)

        # Get TwoTheta and Phi from Q
        theta = np.deg2rad(self.sample.get_two_theta(Q, self.ei.wavelength))
        phi = np.pi / 2.0 - np.deg2rad(self.sample.get_phi(Q))

        # neutron mass in kg
        m_n = neutron_mass * 1e-3

        # Velocity Vector
        vel = np.matrix([[vi, vf]]) * m_n / hbar

        # Appendix A.1 Violini :: spherical detector
        if self.detector.shape == "spherical":
            r = np.matrix([[np.cos(theta_i) * np.cos(phi_i), -np.cos(theta) * np.cos(phi)],
                           [np.sin(theta_i) * np.cos(phi_i), -np.sin(theta) * np.cos(phi)],
                           [np.sin(phi_i),                   -np.sin(phi)]])

            r_tt = np.matrix([[0,  np.sin(theta) * np.cos(phi)],
                              [0, -np.cos(theta) * np.cos(phi)],
                              [0,  0]])


            r_ph = np.matrix([[0,  np.cos(theta) * np.sin(phi)],
                              [0,  np.sin(theta) * np.sin(phi)],
                              [0, -np.cos(phi)]])

        # Appendix A.2 and A.3 Violini :: cylindrical detector
        elif self.detector.shape == "cylindrical":
            # Appendix A.3 Violini :: cylindrical detector with vertical axis
            if ~hasattr(self.detector, "orientation") or self.detector.orientation == "vertical":
                r = np.matrix([[np.cos(theta_i) * np.cos(phi_i), -np.cos(theta)],
                               [np.sin(theta_i) * np.cos(phi_i), -np.sin(theta)],
                               [np.sin(phi_i),                   -np.tan(phi)]])

                r_tt = np.matrix([[0,  np.sin(theta)],
                                  [0, -np.cos(theta)],
                                  [0,  0]])

                r_ph = np.matrix([[0,   0],
                                  [0,   0],
                                  [0, -(1 + np.tan(phi) ** 2)]])

            # Appendix A.2 Violini :: cylindrical detector with horizontal axis
            elif self.detector.orientation == "horizontal":
                # TODO: Complete horizontal cylindrical detector
                r = np.matrix([[np.cos(theta_i) * np.cos(phi_i), -np.tan(theta)],
                               [np.sin(theta_i) * np.cos(phi_i), -np.sin(phi)],
                               [np.sin(phi_i),                   -np.cos(phi)]])

                r_tt = np.matrix([])

                r_ph = np.matrix([])

                raise DetectorError("Horizontal cylindrical detector not yet supported")
            else:
                raise DetectorError("Unsupported cylindrical detector specified: {0}".format(self.detector.orientation))
        else:
            raise DetectorError("Detector shape \"{0}\" not supported".format(self.detector.shape))

        # Detector-shape-independent equations
        r_tt_i = np.matrix([[-np.sin(theta_i) * np.cos(phi_i), 0],
                            [ np.cos(theta_i) * np.cos(phi_i), 0],
                            [ 0,                               0]])

        r_ph_i = np.matrix([[-np.cos(theta_i) * np.sin(phi_i), 0],
                            [-np.sin(theta_i) * np.sin(phi_i), 0],
                            [ np.cos(phi_i),                   0]])

        # Eq 11  Violini :: cov(xi)
        sigma_sq = np.matrix(np.diag([sigma_t ** 2,
                                      sigma_t_md ** 2,
                                      sigma_l_pm ** 2,
                                      sigma_l_ms ** 2,
                                      sigma_l_sd ** 2,
                                      sigma_theta_i ** 2,
                                      sigma_phi_i ** 2,
                                      sigma_theta ** 2,
                                      sigma_phi ** 2]))

        # Eq 11 Violini :: Jacobi J
        q_derivs = np.array([r * np.matrix([[-m_n / hbar * vi / ti, m_n / hbar * vf / tf * l_ms / l_pm]]).T,
                             r * np.matrix([[0, -m_n / hbar * vf / tf]]).T,
                             r * np.matrix([[m_n / hbar / ti, -m_n / hbar * vf / tf * l_ms / (vi * l_pm)]]).T,
                             r * np.matrix([[0, m_n / hbar * vf / tf / vi]]).T,
                             r * np.matrix([[0, m_n / hbar / tf]]).T,
                             r_tt_i * vel.T,
                             r_ph_i * vel.T,
                             r_tt * vel.T,
                             r_ph * vel.T]).T * 1e-10

        # Eqs. 14-18 Violini :: Energy Derivatives
        e_derivs = np.matrix([[-m_n * (l_pm ** 2 / ti ** 3 + l_sd ** 2 / tf ** 3 * l_ms / l_pm),
                                m_n * (l_sd ** 2 / tf ** 3),
                                m_n * (l_pm / ti ** 2 + l_sd ** 2 /  tf ** 3 * ti /  l_pm ** 2 * l_ms),
                               -m_n * (l_sd ** 2 / tf ** 3 * ti / l_pm),
                               -m_n * (l_sd / tf ** 2)]]) / (1e-3 * e)

        # Eq 11 Violini :: Jacobi J
        jacobi = np.matrix(np.zeros((4, sigma_sq.shape[0])))
        jacobi[:3, :] = q_derivs
        jacobi[3, :e_derivs.size] = e_derivs

        # Eq 11 Violini :: cov(Q, hw)
        sigma_qe = jacobi * sigma_sq * jacobi.T

        # M = J^-1
        reso = np.linalg.inv(sigma_qe)

        # Transform from (ki, ki_perp, Qz) to (Q_perp, Q_para, Q_z)
        angle_ki_q = get_angle_ki_Q(ki, kf, self.sample.get_q(Q))

        rot_ki_q = np.matrix(np.eye(4))
        rot_ki_q[:2, :2] = np.matrix([[np.cos(angle_ki_q), -np.sin(angle_ki_q)],
                                      [np.sin(angle_ki_q),  np.cos(angle_ki_q)]])

        # Congruent Transformation T' M T
        res = rot_ki_q.T * reso * rot_ki_q

        # Eq 61 Violini :: resolution prefactor
        fwhm = 0
        X = np.append(Q, W)
        for i in range(4):
            for j in range(4):
                fwhm += res[i, j] * X[i] * X[j]

        r0 = 2 * np.log(2) / fwhm

        return r0, res


    def calc_resolution(self, hkle):
        r"""

        Parameters
        ----------
        hkle

        Returns
        -------

        """
        # TODO: handle multiple points
        r0, reso = self.calc_resolution_in_Q_coords(hkle[:3], hkle[3])

        UBmat = np.eye(4)
        UBmat[:3, :3] = self.sample.UBmatrix

        vec = UBmat[:3, :3].dot(np.array(hkle[:3]))
        ang0 = -np.arctan2(vec[1], vec[0])

        Rot = np.matrix(np.eye(4))
        Rot[:2, :2] = np.matrix([[np.cos(-ang0), -np.sin(-ang0)],
                                 [np.sin(-ang0),  np.cos(-ang0)]])

        res_hkl = (Rot * UBmat) * reso * (Rot * UBmat).T

        self.R0, self.RMS, self.RM = chop(r0), chop(res_hkl), chop(reso)

    def calc_projections(self, hkle):
        r"""

        Parameters
        ----------
        hkle

        Returns
        -------

        """
        # TODO: Calculate the projections (slice & project)
        pass

    def res_volume(self, res):
        return 4 / 3 * np.pi * np.sqrt(1 / np.linalg.det(res))
