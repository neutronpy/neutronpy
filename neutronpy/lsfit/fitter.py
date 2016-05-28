# -*- coding: utf-8 -*-
r"""non-linear least squares fitting
"""
import numbers
import warnings
from collections import namedtuple

import numpy as np
from lmfit import minimize, Parameters

from .tools import residual_wrapper


class Fitter(object):
    u"""Wrapper for LMFIT, which is an extension of scipy.optimize

    Parameters
    ----------
    residuals : func
        The residuals function, see description below.

    deriv : func, optional
        Derivatives function, see description below. If a derivatives
        function is given, user-computed explicit derivatives are automatically
        set for all parameters in the attribute :attr:`parinfo`, but this can
        be changed by the user.

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

    Notes
    -----
    Objects of this class are callable and return the fitted parameters when
    called.

    **Residuals function**
    The residuals function must return a NumPy (dtype='d') array with weighted
    deviations between the model and the data. It takes two arguments:
    a NumPy array containing the parameter values and a reference
    to the attribute :attr:`data` which can be any object containing information
    about the data to be fitted. *E.g.*, a tuple like
    ``(xvalues, yvalues, errors)``.

    In a typical scientific problem the residuals should be weighted so that
    each deviate has a Gaussian sigma of 1.0.  If *x* represents values of the
    independent variable, *y* represents a measurement for each value of *x*,
    and *err* represents the error in the measurements, then the deviates
    could be calculated as follows:

    .. math::

       deviates = (y - f(x)) / err

    where *f* is the analytical function representing the model.
    If *err* are the 1-sigma uncertainties in *y*, then

    .. math::

       \sum deviates^2

    will be the total chi-squared value.  Fitter will minimize this value.
    As described above, the values of *x*, *y* and *err* are
    passed through Fitter to the residuals function via the attribute
    :attr:`data`.

    **Derivatives function**
    The optional derivates function can be used to compute weighted function
    derivatives, which are used in the minimization process.  This can be
    useful to save time, or when the derivative is tricky to evaluate
    numerically.

    The function takes three arguments: a NumPy array containing the parameter
    values, a reference to the attribute :attr:`data` and a list with boolean
    values corresponding with the parameters.
    If a boolean in the list is True, the derivative with respect to the
    corresponding parameter should be computed, otherwise it may be ignored.
    Fitter determines these flags depending on how derivatives are
    specified in item ``side`` of the attribute :attr:`parinfo`, or whether
    the parameter is fixed.

    The function must return a NumPy array with partial derivatives with respect
    to each parameter. It must have shape *(n,m)*, where *n*
    is the number of parameters and *m* the number of data points.

    **Configuration attributes**
    The following attributes can be set by the user to specify a
    Fitter object's behavior.

    **This class was adapted from KMPFIT in the Kapteyn project to work with existing
    unit tests**

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
    """

    def __init__(self, residuals, deriv=None, data=None, params0=None,
                 parinfo=None, ftol=1e-10, xtol=1e-10, gtol=1e-10, epsfcn=None,
                 stepfactor=100.0, covtol=1e-14, maxiter=200, maxfev=None,
                 nofinitecheck=False):

        self._m = 0
        self.result = namedtuple('result', [])
        self.config = namedtuple('config', [])

        self.residuals = residual_wrapper(residuals)
        if deriv is not None:
            self.deriv = residual_wrapper(deriv)
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

    @property
    def parinfo(self):
        r"""A list of dicts with parameter constraints, one dict
        per parameter, or None if not given.

        Each dict can have zero or more items with the following keys
        and values:

         ``'fixed'``: a boolean value, whether the parameter is to be held
         fixed or not. Default: not fixed.

         ``'limits'``: a two-element tuple or list with upper end lower
         parameter limits or  None, which indicates that the parameter is not
         bounded on this side. Default: no limits.

         ``'step'``: the step size to be used in calculating the numerical
         derivatives. Default: step size is computed automatically.

         ``'side'``: the sidedness of the finite difference when computing
         numerical derivatives.  This item can take four values:

            * 0 - one-sided derivative computed automatically (default)

            * 1 - one-sided derivative :math:`(f(x+h) - f(x)  )/h`

            * -1 - one-sided derivative :math:`(f(x)   - f(x-h))/h`

            * 2 - two-sided derivative :math:`(f(x+h) - f(x-h))/2h`

            * 3 - user-computed explicit derivatives

            where :math:`h` is the value of the parameter ``'step'``
            described above.

            The "automatic" one-sided derivative method will chose a
            direction for the finite difference which does not
            violate any constraints.  The other methods do not
            perform this check.  The two-sided method is in
            principle more precise, but requires twice as many
            function evaluations.  Default: 0.

         ``'deriv_debug'``: boolean to specify console debug logging of
         user-computed derivatives. True: enable debugging.
         If debugging is enabled,
         then ``'side'`` should be set to 0, 1, -1 or 2, depending on which
         numerical derivative you wish to compare to.
         Default: False.

        As an example, consider a function with four parameters of which the
        first parameter should be fixed and for the third parameter explicit
        derivatives should be used. In this case, ``parinfo`` should have the
        value ``[{'fixed': True}, None, {'side': 3}, None]`` or
        ``[{'fixed': True}, {}, {'side': 3}, {}]``.

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
        r"""The fitted parameters. This attribute has the same type as :attr:`params0`.
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
        r"""Finite derivative step size. Default: 2.2204460e-16
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
        r"""(DEPRECIATED) Range tolerance for covariance calculation. Default: 1e-14
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
        """Standard errors estimated from :math:`\sqrt{diag(covar) * \chi^{2}_{reduced}`
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

        self.result.orignorm = np.sum(self.residuals(params0, self.data) ** 2)

        result = minimize(self.residuals, p, Dfun=self.deriv, method='leastsq', ftol=self.ftol, xtol=self.xtol, gtol=self.gtol,
                          maxfev=self.maxfev, epsfcn=self.epsfcn, factor=self.stepfactor, args=(self.data,), kws=None)

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
