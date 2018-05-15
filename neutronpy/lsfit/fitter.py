# -*- coding: utf-8 -*-
r"""non-linear least squares fitting
"""
import numbers
import warnings
from collections import namedtuple

import numpy as np

from lmfit import Minimizer, Parameters, minimize

from .plot import PlotFit
from .tools import residual_wrapper


class Fitter(PlotFit):
    u"""Wrapper for LMFIT, which is a high-level extension for
    scipy.optimize.leastsq. Performs Non-Linear Least Squares fitting using
    the Levenberg-Marquardt method.

    Parameters
    ----------
    residuals : func
        The residuals function, see description below.

    derivatives : func, optional
        Derivatives function, to compute the Jacobian of the residuals
        function with derivatives across the rows. If this is None, the
        Jacobian will be estimated.

    data : tuple, optional
        Default: None

    params0 : list, optional
        Default: None

    parinfo : list, optional
        Default: None

    ftol : float, optional
        Default: 1e-10

    xtol : float, optional
        Default: 1e-10

    epsfcn : float, optional
        Default: 2.2204460492503131e-16

    stepfactor : float, optional
        Default: 100.0

    covtol : float, optional
        Default: 1e-14

    maxiter : int, optional
        Default: 200

    maxfev : int, optional
        Default: 0

    nofinitecheck : bool, optional
        Default: False

    nan_policy : str, optional
        Default: 'omit'. Determines how NaN values are handled: 'raise',
        'propagate' or 'omit'.

    Notes
    -----
    Objects of this class are callable, returning the fitted parameters.

    **Residuals function**
    The residuals function must return an ndarray with weighted deviations
    between the model and the data. It takes two arguments, a list of the
    parameter values and a reference for the attribute :attr:`data`, a tuple
    e.g. ``(x, y, err)``. In a typical scientific problem the residuals should
    be weighted so that each deviate has a Gaussian sigma of 1.0.  If ``x``
    represents the independent variable, ``y`` represents an intensity for
    each value of ``x``, and ``err`` represents the error, then the deviates
    could be calculated as follows:

    .. math::

       d = (y - f(x)) / err

    where *f* is the model function. If *err* are 1-sigma uncertainties in
    ``y``, then

    .. math::

       \sum d^2

    is the total chi-squared.  :py:meth:`Fitter.fit` will minimize this value.
    ``x``, ``y`` and ``err`` are passed to the residuals function from
    :attr:`data`.

    Attributes
    ----------
    parinfo
    params0
    data
    ftol
    xtol
    gtol
    epsfcn
    stepfactor
    covtol
    maxiter
    maxfev
    params
    xerror
    covar
    chi2_min
    orignorm
    rchi2_min
    stderr
    npar
    nfree
    npegged
    dof
    resid
    niter
    nfev
    status
    message
    residuals
    nofinitecheck

    Methods
    -------
    fit
    plot
    build_param_table
    __call__
    """

    def __init__(self, residuals, derivatives=None, data=None, params0=None, parinfo=None, ftol=1e-10, xtol=1e-10,
                 gtol=1e-10, epsfcn=None, stepfactor=100.0, covtol=1e-14, maxiter=200, maxfev=None,
                 nofinitecheck=False, nan_policy='omit'):

        self._m = 0
        self.result = namedtuple('result', [])
        self.config = namedtuple('config', [])

        self.residuals = residual_wrapper(residuals)
        if derivatives is not None:
            self.deriv = residual_wrapper(derivatives)
        else:
            self.deriv = None

        self.data = data
        self.params0 = params0
        self.parinfo = parinfo
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.epsfcn = epsfcn
        self.stepfactor = stepfactor
        self.covtol = covtol
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.nofinitecheck = nofinitecheck
        self.nan_policy = nan_policy

    def __call__(self, params0=None):
        if hasattr(self, 'params'):
            return self.params
        elif self.params0 is not None and params0 is None:
            self.fit(self.params0)
            return self.params
        elif params0 is not None:
            self.fit(params0)
            return self.params
        else:
            raise ValueError('params0 is undefined, no fit can be performed')

    @property
    def parinfo(self):
        r"""A list of dicts with parameter constraints, one dict per
        parameter, or None if not given.

        Each dict can have zero or more items with the following keys and
        values:

         ``'fixed'``: bool
            Parameter to be fixed. Default: False.

         ``'limits'``: list
            Two-element list with upper end lower parameter limits or None,
            which indicates that the parameter is not bounded on this side.
            Default: None.
        """
        return self._parinfo

    @parinfo.setter
    def parinfo(self, value):
        if isinstance(value, (list, tuple)):
            if np.all([isinstance(item, (type(None), dict)) for item in value]):
                self._parinfo = value
            else:
                raise ValueError
        elif value is None:
            self._parinfo = None
        else:
            raise ValueError

    @property
    def params(self):
        r"""The fitted parameters. This attribute has the same type as
        :attr:`params0`.
        """
        return self.result.params

    @property
    def params0(self):
        r"""Required attribute. A NumPy array, a tuple or a list with the
        initial parameters values.
        """
        return self._params0

    @params0.setter
    def params0(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            self._params0 = value
        elif value is None:
            self._params0 = None
        else:
            raise ValueError

    @property
    def data(self):
        r"""Required attribute. Python object with information for the
        residuals function and the derivatives function. See above.
        """
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, tuple):
            self._data = value
        else:
            ValueError

    @property
    def deriv(self):
        return self._deriv

    @deriv.setter
    def deriv(self, value):
        self._deriv = value

    @property
    def ftol(self):
        r"""Relative :math:`\chi^2` convergence criterium. Default: 1e-10
        """
        return self.config.ftol

    @ftol.setter
    def ftol(self, value):
        if isinstance(value, numbers.Number):
            self.config.ftol = value
        else:
            raise ValueError

    @property
    def xtol(self):
        r"""Relative parameter convergence criterium. Default: 1e-10
        """
        return self.config.xtol

    @xtol.setter
    def xtol(self, value):
        if isinstance(value, numbers.Number):
            self.config.xtol = value
        else:
            raise ValueError

    @property
    def gtol(self):
        r"""Orthogonality convergence criterium. Default: 1e-10
        """
        return self.config.gtol

    @gtol.setter
    def gtol(self, value):
        if isinstance(value, numbers.Number):
            self.config.gtol = value
        else:
            raise ValueError

    @property
    def epsfcn(self):
        r"""Finite derivative step size. Default: 2.2204460492503131e-16
        """
        return self.config.epsfcn

    @epsfcn.setter
    def epsfcn(self, value):
        if value is None:
            value = np.finfo(np.float64).eps
        if isinstance(value, numbers.Number):
            self.config.epsfcn = value
        else:
            raise ValueError

    @property
    def stepfactor(self):
        r"""Initial step bound. Default: 100.0
        """
        return self.config.stepfactor

    @stepfactor.setter
    def stepfactor(self, value):
        if isinstance(value, numbers.Number):
            self.config.stepfactor = value
        else:
            raise ValueError

    @property
    def covtol(self):
        r"""(DEPRECIATED) Range tolerance for covariance calculation.
        Default: 1e-14
        """
        warnings.warn('covtol is Depreciated and has no effect', DeprecationWarning)
        return self.config.covtol

    @covtol.setter
    def covtol(self, value):
        if isinstance(value, numbers.Number):
            self.config.covtol = value
        else:
            raise ValueError

    @property
    def maxiter(self):
        r"""(DEPRECIATED) Maximum number of iterations. Default: 200
        """
        warnings.warn('maxiter is Depreciated and has no effect', DeprecationWarning)
        return self.config.maxiter

    @maxiter.setter
    def maxiter(self, value):
        if isinstance(value, int):
            self.config.maxiter = value
        else:
            raise ValueError

    @property
    def maxfev(self):
        r"""Maximum number of function evaluations. Default: 0
        """
        return self.config.maxfev

    @maxfev.setter
    def maxfev(self, value):
        if isinstance(value, int):
            self.config.maxfev = value
        elif value is None:
            self.config.maxfev = 0
        else:
            raise ValueError

    @property
    def nofinitecheck(self):
        r"""(DEPRECIATED) Does not check for finite values. Default: False
        """
        warnings.warn('nofinitecheck is Depreciated and has no effect', DeprecationWarning)
        return self._nofinitecheck

    @nofinitecheck.setter
    def nofinitecheck(self, value):
        if isinstance(value, bool):
            self.config.nofinitecheck = value
        else:
            raise ValueError

    @property
    def nan_policy(self):
        r"""Determines how NaN's are handled by minimizer. Default: 'omit'
        """
        return self._nan_policy

    @nan_policy.setter
    def nan_policy(self, value):
        if isinstance(value, str):
            self._nan_policy = value
        else:
            raise ValueError

    @property
    def npar(self):
        r"""Number of parameters
        """
        try:
            return len(self.params0)
        except TypeError:
            return None

    @property
    def message(self):
        """Success/error message
        """
        return self.result.message

    @property
    def chi2_min(self):
        """Final :math:`\chi^2`
        """
        return self.result.bestnorm

    @property
    def orignorm(self):
        """Initial :math:`\chi^2`.
        """
        return self.result.orignorm

    @property
    def niter(self):
        """Number of iterations
        """
        return self.result.niter

    @property
    def nfev(self):
        """Number of function evaluations
        """
        return self.result.nfev

    @property
    def status(self):
        """Status code of fit passed from scipy.optimize.leastsq
        """
        return self.result.status

    @property
    def nfree(self):
        """Number of free parameters
        """
        return self.result.nfree

    @property
    def npegged(self):
        """Number of fixed parameters
        """
        return self.result.npegged

    @property
    def covar(self):
        """Parameter covariance matrix
        """
        return self.result.covar

    @property
    def resid(self):
        """Residuals
        """
        return self.result.resid

    @property
    def xerror(self):
        """Parameter uncertainties (:math:`1 \sigma`)
        """
        return self.result.xerror

    @property
    def dof(self):
        """Degrees of freedom
        """
        return self._m - self.nfree

    @property
    def rchi2_min(self):
        """Minimum reduced :math:`\chi^2`.
        """
        return self.result.redchi

    @property
    def stderr(self):
        """Standard errors estimated from
        :math:`\sqrt{diag(covar) * \chi^{2}_{reduced}`
        """
        return np.sqrt(np.diagonal(self.covar) * self.rchi2_min)



    def fit(self, params0):
        r"""Perform a fit with the provided parameters.

        Parameters
        ----------
        params0 : list
            Initial fitting parameters

        """
        self.params0 = params0
        p = Parameters()

        if self.parinfo is None:
            self.parinfo = [None] * len(self.params0)
        else:
            assert (len(self.params0) == len(self.parinfo))

        for i, (p0, parin) in enumerate(zip(self.params0, self.parinfo)):
            p.add(name='p{0}'.format(i), value=p0)

            if parin is not None:
                if 'limits' in parin:
                    p['p{0}'.format(i)].set(min=parin['limits'][0])
                    p['p{0}'.format(i)].set(max=parin['limits'][1])
                if 'fixed' in parin:
                    p['p{0}'.format(i)].set(vary=not parin['fixed'])

        if np.all([not value.vary for value in p.values()]):
            raise Exception('All parameters are fixed!')

        self.lmfit_minimizer = Minimizer(self.residuals, p, nan_policy=self.nan_policy, fcn_args=(self.data,))

        self.result.orignorm = np.sum(self.residuals(params0, self.data) ** 2)

        result = self.lmfit_minimizer.minimize(Dfun=self.deriv, method='leastsq', ftol=self.ftol,
                                               xtol=self.xtol, gtol=self.gtol, maxfev=self.maxfev, epsfcn=self.epsfcn,
                                               factor=self.stepfactor)

        self.result.bestnorm = result.chisqr
        self.result.redchi = result.redchi
        self._m = result.ndata
        self.result.nfree = result.nfree
        self.result.resid = result.residual
        self.result.status = result.ier
        self.result.covar = result.covar
        self.result.xerror = [result.params['p{0}'.format(i)].stderr for i in range(len(result.params))]

        self.result.params = [result.params['p{0}'.format(i)].value for i in range(len(result.params))]

        self.result.message = result.message

        self.lmfit_result = result

        if not result.errorbars or not result.success:
            warnings.warn(self.result.message)

        return result.success
