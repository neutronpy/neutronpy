# -*- coding: utf-8 -*-
import numpy as np
from ..constants import BOLTZMANN_IN_MEV_K
from ..energy import Energy


class Analysis(object):
    r"""Class containing methods for the Data class

    Attributes
    ----------
    detailed_balance_factor

    Methods
    -------
    integrate
    position
    width
    scattering_function
    dynamic_susceptibility
    estimate_background

    """
    @property
    def detailed_balance_factor(self):
        r"""Returns the detailed balance factor (sometimes called the Bose
        factor)

        Parameters
        ----------
            None

        Returns
        -------
        dbf : ndarray
            The detailed balance factor (temperature correction)

        """

        return 1. - np.exp(-self.Q[:, 3] / BOLTZMANN_IN_MEV_K / self.temp)

    def integrate(self, background=None, **kwargs):
        r"""Returns the integrated intensity within given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : float
            The integrated intensity either over all data, or within
            specified boundaries

        """
        if background is not None:
            background = self.estimate_background(background)
        else:
            background = 0

        result = 0
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for i in range(4):
                result += np.trapz(self.intensity[to_fit] - background,
                                   x=self.Q[to_fit, i])
        else:
            for i in range(4):
                result += np.trapz(self.intensity - background,
                                   x=self.Q[:, i])

        return result

    def position(self, background=None, **kwargs):
        r"""Returns the position of a peak within the given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : tup
            The result is a tuple with position in each dimension of Q,
            (h, k, l, e)

        """
        if background is not None:
            background = self.estimate_background(background)
        else:
            background = 0

        result = ()
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = self.Q[to_fit, j] * (self.intensity[to_fit] - background)
                    _result += np.trapz(y, x=self.Q[to_fit, i]) / self.integrate(**kwargs)
                result += (_result,)
        else:
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = self.Q[:, j] * (self.intensity - background)
                    _result += np.trapz(y, x=self.Q[:, i]) / self.integrate(**kwargs)
                result += (_result,)

        return result

    def width(self, background=None, **kwargs):
        r"""Returns the mean-squared width of a peak within the given bounds

        Parameters
        ----------
        bounds : bool, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : tup
            The result is a tuple with the width in each dimension of Q,
            (h, k, l, e)

        """
        if background is not None:
            background = self.estimate_background(background)
        else:
            background = 0

        result = ()
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = (self.Q[to_fit, j] - self.position(**kwargs)[j]) ** 2 * \
                        (self.intensity[to_fit] - background)
                    _result += np.trapz(y, x=self.Q[to_fit, i]) / self.integrate(**kwargs)
                result += (_result,)
        else:
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = (self.Q[:, j] - self.position(**kwargs)[j]) ** 2 * \
                        (self.intensity - background)
                    _result += np.trapz(y, x=self.Q[:, i]) / self.integrate(**kwargs)
                result += (_result,)

        return result

    def scattering_function(self, material, ei):
        r"""Returns the neutron scattering function, i.e. the detector counts
        scaled by :math:`4 \pi / \sigma_{\mathrm{tot}} * k_i/k_f`.

        Parameters
        ----------
        material : object
            Definition of the material given by the :py:class:`.Material`
            class

        ei : float
            Incident energy in meV

        Returns
        -------
        counts : ndarray
            The detector counts scaled by the total scattering cross section
            and ki/kf
        """
        ki = Energy(energy=ei).wavevector
        kf = Energy(energy=ei - self.e).wavevector

        return (4 * np.pi / (material.total_scattering_cross_section) * ki /
                kf * self.detector)

    def dynamic_susceptibility(self, material, ei):
        r"""Returns the dynamic susceptibility
        :math:`\chi^{\prime\prime}(\mathbf{Q},\hbar\omega)`

        Parameters
        ----------
        material : object
            Definition of the material given by the :py:class:`.Material`
            class

        ei : float
            Incident energy in meV

        Returns
        -------
        counts : ndarray
            The detector counts turned into the scattering function multiplied
            by the detailed balance factor
        """
        return (self.scattering_function(material, ei) *
                self.detailed_balance_factor)

    def estimate_background(self, bg_params):
        r"""Estimate the background according to ``type`` specified.

        Parameters
        ----------
        bg_params : dict
            Input dictionary has keys 'type' and 'value'. Types are
                * 'constant' : background is the constant given by 'value'
                * 'percent' : background is estimated by the bottom x%, where x
                  is value
                * 'minimum' : background is estimated as the detector counts

        Returns
        -------
        background : float or ndarray
            Value determined to be the background. Will return ndarray only if
            `'type'` is `'constant'` and `'value'` is an ndarray
        """
        if bg_params['type'] == 'constant':
            return bg_params['value']

        elif bg_params['type'] == 'percent':
            inten = self.intensity[self.intensity >= 0.]
            Npts = int(inten.size * (bg_params['value'] / 100.))
            min_vals = inten[np.argsort(inten)[:Npts]]
            background = np.average(min_vals)
            return background

        elif bg_params['type'] == 'minimum':
            return min(self.intensity)

        else:
            return 0
