r"""
=============
Module kmpfit
=============

.. author:: Hans Terlouw <gipsy@astro.rug.nl>
.. highlight:: python
   :linenothreshold: 5

Introduction
------------
This module provides the class Fitter, which uses the implementation in
C of `MPFIT <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_,
Craig Markwardt's non-linear least squares curve fitting routines for
IDL.  MPFIT uses the Levenberg-Marquardt technique to solve the
least-squares problem, which is a particular strategy for iteratively
searching for the best fit.  In its typical use, MPFIT will be used to
fit a user-supplied function (the "model") to user-supplied data points
(the "data") by adjusting a set of parameters.  MPFIT is based upon the
robust routine MINPACK-1 (LMDIF.F) by Mor\u00e9 and collaborators.

For example, a researcher may think that a set of observed data
points is best modelled with a Gaussian curve.  A Gaussian curve is
parameterized by its mean, standard deviation and normalization.
MPFIT will, within certain constraints, find the set of parameters
which best fits the data.  The fit is "best" in the least-squares
sense; that is, the sum of the weighted squared differences between
the model and data is minimized.

.. seealso::
   :doc:`Tutorial<kmpfittutorial>` with background information for this module
   and practical examples.

Class Fitter
------------
.. autoclass:: Fitter(residuals, deriv=None, ...)

Testing derivatives
...................

In principle, the process of computing explicit derivatives should be
straightforward.  In practice, the computation can be error prone,
often being wrong by a sign or a scale factor.

In order to be sure that the explicit derivatives are correct,
for debugging purposes the
user can set the attribute parinfo['deriv_debug'] = True
for any parameter. This will cause :meth:`Fitter.fit` to
print *both* explicit derivatives and numerical derivatives to the
console so that the user can compare the results.

When debugging derivatives, it is important to set parinfo['side']
to the kind of numerical derivative to compare with:
it should be set to 0, 1, -1, or 2, and *not* set to 3.
When parinfo['deriv_debug'] is set for a parameter, then
:meth:`Fitter.fit` automatically understands to request user-computed derivatives.

The console output will be sent to the standard output, and will
appear as a block of ASCII text like this::

  FJAC DEBUG BEGIN
  # IPNT FUNC DERIV_U DERIV_N DIFF_ABS DIFF_REL
  FJAC PARM 1
  ....  derivative data for parameter 1 ....
  FJAC PARM 2
  ....  derivative data for parameter 2 ....
  ....  and so on ....
  FJAC DEBUG END

which is to say, debugging data will be bracketed by pairs of "FJAC
DEBUG" BEGIN/END phrases.  Derivative data for individual parameter i
will be labeled by "FJAC PARM i".  The columns are, in order,

  IPNT - data point number :math:`j`

  FUNC - residuals function evaluated at :math:`x_j`

  DERIV_U - user-calculated derivative
  :math:`{\\partial f(x_j)}/{\\partial p_i}`

  DERIV_N - numerically calculated derivative according to the value of
  parinfo['side']

  DIFF_ABS - difference between DERIV_U and DERIV_N: fabs(DERIV_U-DERIV_N)

  DIFF_REL - relative difference: fabs(DERIV_U-DERIV_N)/DERIV_U

Since individual numerical derivative values may contain significant
round-off errors, it is up to the user to critically compare DERIV_U
and DERIV_N, using DIFF_ABS and DIFF_REL as a guide.



Example
.......

.. code-block:: python
    #!/usr/bin/env python

    import numpy
    from kapteyn import kmpfit

    def residuals(p, d):
       a, b, c = p
       x, y, w = d
       return (y - (a*x*x+b*x+c))/w

    x = numpy.arange(-50,50,0.2)
    y = 2*x*x + 3*x - 3 + 2*numpy.random.standard_normal(x.shape)
    w = numpy.ones(x.shape)

    a = [x, y, w]
    f = kmpfit.Fitter(residuals, params0=[1, 2, 0], data=a)

    f.fit()                                     # call fit method
    print f.params
    print f.message
    # result:
    # [2.0001022845514451, 3.0014019147386, -3.0096629062273133]
    # mpfit (potential) success: Convergence in chi-square value (1)

    a[1] = 3*x*x  - 2*x - 5 + 0.5*numpy.random.standard_normal(x.shape)
    print f(params0=[2, 0, -1])                 # call Fitter object
    # result:
    # [3.0000324686457871, -1.999896340813663, -5.0060187435412962]

"""
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
from libc.stdlib cimport calloc, free
from kmpfit cimport *
cimport numpy as np

np.import_array()

MP_OK = {1: 'Convergence in chi-square value',
         2: 'Convergence in parameter value',
         3: 'Convergence in chi-square and parameter value',
         4: 'Convergence in orthogonality',
         5: 'Maximum number of iterations reached',
         6: 'ftol is too small; no further improvement',
         7: 'xtol is too small; no further improvement',
         8: 'gtol is too small; no further improvement'}

MP_ERR = {0: 'General input parameter error',
          -16: 'User function produced non-finite values',
          -17: 'No user function was supplied',
          -18: 'No user data points were supplied',
          -19: 'No free parameters',
          -20: 'Memory allocation error',
          -21: 'Initial values inconsistent w constraints',
          -22: 'Initial constraints inconsistent',
          -23: 'General input parameter error',
          -24: 'Not enough degrees of freedom'}


cdef int xmpfunc(int *mp, int n, double *x, double **fvecp, double **dvec, void *private_data) except -1:
    cdef double *e
    cdef double *f
    cdef double *y
    cdef double *fvec
    cdef double *d
    cdef double *cjac
    cdef int i, j, m
    cdef np.npy_intp *shape = [n]

    self = < Fitter > private_data
    for i in range(n):
        if x[i] != x[i]:  # not finite?
            self.message = 'Non-finite parameter from mpfit.c'
            raise ValueError(self.message)
    p = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, x)
    deviates = self.residuals(p, self.data)

    f = < double* > np.PyArray_DATA(deviates)
    if mp[0]:
        m = mp[0]
        fvec = fvecp[0]
        for i in range(m):
            fvec[i] = f[i]
    else:
        fvecp[0] = f
        mp[0] = deviates.size
        self.m = mp[0]
        self.deviates = deviates  # keep a reference to protect from GC

        self.allocres()

    if dvec != NULL and self.deriv is not None:
        for i in range(n):
            self.dflags[i] = bool(< int > dvec[i])
        jac = self.deriv(p, self.data, self.dflags)
        cjac = < double* > np.PyArray_DATA(jac)
        for j in range(n):
            d = dvec[j]
            if d != NULL:
                for i in range(m):
                    d[i] = cjac[j * m + i]

    return 0


cdef class Fitter(object):
    """Uses the implementation in
    C of `MPFIT <http://www.physics.wisc.edu/~craigm/idl/cmpfit.html>`_,
    Craig Markwardt's non-linear least squares curve fitting routines for
    IDL.  MPFIT uses the Levenberg-Marquardt technique to solve the
    least-squares problem, which is a particular strategy for iteratively
    searching for the best fit.  In its typical use, MPFIT will be used to
    fit a user-supplied function (the "model") to user-supplied data points
    (the "data") by adjusting a set of parameters.  MPFIT is based upon the
    robust routine MINPACK-1 (LMDIF.F) by Mor\u00e9 and collaborators.

    Parameters
    ----------
    residuals : func
          the residuals function, see description below.
    deriv : func
          optional derivatives function, see description below. If a derivatives
          function is given, user-computed explicit derivatives are automatically
          set for all parameters in the attribute :attr:`parinfo`, but this can
          be changed by the user.
    kwargs :
          other parameters, each corresponding with one of the configuration
          attributes described below. They can be defined here, when the Fitter
          object is created, or later. The attributes :attr:`params0` and
          :attr:`data` must be defined before the method :meth:`fit` is
          called.

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
    version
    status
    message
    residuals
    nofinitecheck

    Methods
    -------
    fit
    confidence_band
    """

    cdef public object _parinfo  # parinfo
    cdef mp_par *c_pars  # parinfo: C-representation
    cdef int m
    cdef mp_config *config
    cdef mp_result *result
    cdef public object _params0  # initial fitting parameters
    cdef object params_t  # parameter type
    cdef double *xall  # parameters: C-representation
    cdef readonly int _npar  # number of parameters

    cdef public object residuals, _data  # residuals function, private data
    cdef object deriv, dflags  # derivatives function, flags

    cdef object deviates  # deviates
    cdef readonly object _message  # status message

    def __cinit__(self):
        self.config = < mp_config* > calloc(1, sizeof(mp_config))
        self.result = < mp_result* > calloc(1, sizeof(mp_result))

    def __dealloc__(self):
        free(self.config)
        free(self.result.resid)
        free(self.result.xerror)
        free(self.result.covar)
        free(self.result)
        free(self.c_pars)
        free(self.xall)

    def __init__(self, residuals, deriv=None, params0=None, parinfo=None, ftol=None, xtol=None,
                 gtol=None, epsfcn=None, stepfactor=None, covtol=None, maxiter=None, maxfev=None,
                 nofinitecheck=None, data=None):
        self.npar = 0
        self.m = 0
        self.residuals = residuals  # residuals function
        self.deriv = deriv  # derivatives function
        self.params0 = params0  # initial fitting parameters
        self.parinfo = parinfo  # parameter constraints
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.epsfcn = epsfcn
        self.stepfactor = stepfactor
        self.covtol = covtol
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.nofinitecheck = nofinitecheck
        self.data = data  # args to residuals function

    property parinfo:
        """A list of dicts with parameter contraints, one dict
        per parameter, or None if not given.

        Each dict can have zero or more items with the following keys
        and values:

         ``'fixed'``: a boolean value, whether the parameter is to be held fixed or
         not. Default: not fixed.

         ``'limits'``: a two-element tuple or list with upper end lower parameter
         limits or  None, which indicates that the parameter is not bounded on
         this side. Default: no limits.

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
        derivatives should be used. In this case, ``parinfo`` should have the value
        ``[{'fixed': True}, None, {'side': 3}, None]`` or
        ``[{'fixed': True}, {}, {'side': 3}, {}]``.

        """
        def __get__(self):
            return self._parinfo

        def __set__(self, value):
            if value is not None:
                self._parinfo = value

    property params:
        '''A NumPy array, list or tuple with the fitted parameters. This attribute has the same type as :attr:`params0`.
        '''
        def __get__(self):
            cdef np.npy_intp *shape = [self.npar]
            value = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.xall).copy()
            if self.params_t is not None:
                return self.params_t(value)
            else:
                return value

        def __set__(self, value):
            if value is None:
                return
            cdef int i, l
            cdef double *xall
            if not isinstance(value, np.ndarray):
                self.params_t = type(value)
                l = len(value)
            else:
                l = value.size
            if self.npar == 0:
                self.npar = l
            elif l != self.npar:
                self.message = 'inconsistent parameter array size'
                raise ValueError(self.message)
            xall = < double* > calloc(self.npar, sizeof(double))
            for i in range(self.npar):
                xall[i] = value[i]
            free(self.xall)
            self.xall = xall
            if self.dflags is None:
                self.dflags = [False] * self.npar  # flags for deriv()
            if self.parinfo is None:
                if self.deriv is not None:
                    self.parinfo = [{'side': 3}] * self.npar
                else:
                    self.parinfo = [None] * self.npar

    property params0:
        '''Required attribute. A NumPy array, a tuple or a list with the initial parameters values.
        '''
        def __get__(self):
            return self._params0

        def __set__(self, value):
            if value is not None:
                self._params0 = value

    property data:
        '''Required attribute. Python object with information for the residuals function and the derivatives function. See above.
        '''
        def __get__(self):
            return self._data

        def __set__(self, value):
            self._data = value

    property ftol:
        '''Relative :math:`\chi^2` convergence criterium. Default: 1e-10
        '''
        def __get__(self):
            return self.config.ftol

        def __set__(self, value):
            if value is not None:
                self.config.ftol = value

        def __del__(self):
            self.config.ftol = 0.0

    property xtol:
        '''Relative parameter convergence criterium. Default: 1e-10
        '''
        def __get__(self):
            return self.config.xtol

        def __set__(self, value):
            if value is not None:
                self.config.xtol = value

        def __del__(self):
            self.config.xtol = 0.0

    property gtol:
        '''Orthogonality convergence criterium. Default: 1e-10
        '''
        def __get__(self):
            return self.config.gtol

        def __set__(self, value):
            if value is not None:
                self.config.gtol = value
        def __del__(self):
            self.config.gtol = 0.0

    property epsfcn:
        '''Finite derivative step size. Default: 2.2204460e-16 (MACHEP0)
        '''
        def __get__(self):
            return self.config.epsfcn

        def __set__(self, value):
            if value is not None:
                self.config.epsfcn = value

        def __del__(self):
            self.config.epsfcn = 0.0

    property stepfactor:
        '''Initial step bound. Default: 100.0
        '''
        def __get__(self):
            return self.config.stepfactor

        def __set__(self, value):
            if value is not None:
                self.config.stepfactor = value

        def __del__(self):
            self.config.stepfactor = 0.0

    property covtol:
        '''Range tolerance for covariance calculation. Default: 1e-14
        '''
        def __get__(self):
            return self.config.covtol

        def __set__(self, value):
            if value is not None:
                self.config.covtol = value

        def __del__(self):
            self.config.covtol = 0.0

    property maxiter:
        '''Maximum number of iterations. Default: 200
        '''
        def __get__(self):
            return self.config.maxiter

        def __set__(self, value):
            if value is not None:
                self.config.maxiter = value

        def __del__(self):
            self.config.maxiter = 0

    property nofinitecheck:
        r'''Does not check for finite values. Default: None
        '''
        def __get__(self):
            return self.config.nofinitecheck

        def __set__(self, value):
            if value is not None:
                self.config.nofinitecheck = value

    property maxfev:
        '''Maximum number of function evaluations. Default: 0 (no limit)
        '''
        def __get__(self):
            return self.config.maxfev

        def __set__(self, value):
            if value is not None:
                self.config.maxfev = value

        def __del__(self):
            self.config.maxfev = 0

    property npar:
        '''Number of parameters.
        '''
        def __get__(self):
            return self._npar

        def __set__(self, value):
            if value is not None:
                self._npar = value

    property message:
        '''Message string.
        '''
        def __get__(self):
            return self._message

        def __set__(self, value):
            self._message = value

    @property
    def chi2_min(self):
        '''Final :math:`\chi^2`.
        '''
        return self.result.bestnorm

    @property
    def orignorm(self):
        '''Starting value of :math:`\chi^2`.
        '''
        return self.result.orignorm

    @property
    def niter(self):
        '''Number of iterations.
        '''
        return self.result.niter

    @property
    def nfev(self):
        '''Number of function evaluations.
        '''
        return self.result.nfev

    @property
    def status(self):
        '''Fitting status code.
        '''
        return self.result.status

    @property
    def nfree(self):
        '''Number of free parameters.
        '''
        return self.result.nfree

    @property
    def npegged(self):
        '''Number of pegged parameters.
        '''
        return self.result.npegged

    @property
    def version(self):
        '''mpfit.c's version string.
        '''
        return self.result.version

    @property
    def covar(self):
        '''Final parameter covariance (NumPy-) matrix.
        '''
        cdef np.npy_intp *shape = [self.npar, self.npar]
        value = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, self.result.covar).copy()
        return np.matrix(value)

    @property
    def resid(self):
        '''Final residuals.
        '''
        cdef np.npy_intp *shape = [self.m]
        value = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.result.resid).copy()
        return value

    @property
    def xerror(self):
        '''Final parameter uncertainties (:math:`1 \sigma`)
        '''
        cdef np.npy_intp *shape = [self.npar]
        value = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.result.xerror).copy()
        return value

    @property
    def dof(self):
        '''Number of degrees of freedom.
        '''
        return self.m - self.nfree

    @property
    def rchi2_min(self):
        '''Minimum reduced :math:`\chi^2`.
        '''
        return self.chi2_min / self.dof

    @property
    def stderr(self):
        '''Standard errors.
        '''
        return np.sqrt(np.diagonal(self.covar) * self.rchi2_min)

    cdef allocres(self):
        # allocate arrays in mp_result_struct
        free(self.result.resid)
        self.result.resid = < double* > calloc(self.m, sizeof(double))
        free(self.result.xerror)
        self.result.xerror = < double* > calloc(self.npar, sizeof(double))
        free(self.result.covar)
        self.result.covar = < double* > calloc(self.npar * self.npar, sizeof(double))

    def fit(self, params0=None):
        """Perform a fit with the current values of parameters and other attributes.

        Parameters
        ----------
        params0 : array_like
            Optional argument *params0*: initial fitting parameters.
            (Default: previous initial values are used.)

        """
        cdef mp_par *c_par

        if params0 is not None:
            self.params0 = params0
        if self.params0 is None:
            self.message = 'no initial fitting parameters specified'
            raise RuntimeError(self.message)
        else:
            self.params = self.params0

        if len(self.parinfo) != self.npar:
            self.message = 'inconsistent parinfo list length'
            raise ValueError(self.message)
        if self.c_pars == NULL:
            self.c_pars = < mp_par* > calloc(self.npar, sizeof(mp_par))
        for ipar, par in enumerate(self.parinfo):
            c_par = & self.c_pars[ipar]

            try:
                c_par.fixed = par['fixed']
            except:
                c_par.fixed = 0

            try:
                limits = par['limits']
                for limit in (0, 1):
                    if limits[limit] is not None:
                        c_par.limited[limit] = 1
                        c_par.limits[limit] = limits[limit]
            except:
                for limit in (0, 1):
                    c_par.limited[limit] = 0
                    c_par.limits[limit] = 0.0

            try:
                c_par.step = par['step']
            except:
                c_par.step = 0

            try:
                c_par.side = par['side']
            except:
                c_par.side = 0

            try:
                c_par.deriv_debug = par['deriv_debug']
            except:
                c_par.deriv_debug = 0

        status = mpfit(< mp_func > xmpfunc, self.npar, self.xall, self.c_pars, self.config, < void* > self, self.result)

        if status <= 0:
            if status in MP_ERR:
                self.message = 'mpfit error: {0:s} ({1:d})'.format(MP_ERR[status], status)
            else:
                self.message = 'mpfit error, status={0:d}'.format(status)

            raise RuntimeError(self.message)

        if status in MP_OK:
            self.message = 'mpfit (potential) success: {0:s} ({1:d})'.format(MP_OK[status], status)
        else:
            self.message = None

        return status

    def __call__(self, params0=None):
        self.fit(params0)
        return self.params

    def confidence_band(self, x, dfdp, confprob, f, abswei=False):
        """After the method :meth:`fit` has been called, this method calculates
        the upper and lower value of the confidence interval for all elements
        of the NumPy array *x*. The model values and
        the arrays with confidence limits are returned and can be used to
        plot confidence bands.

        Parameters
        ----------
        x : ndarray
           NumPy array with the independent values for which the confidence interval
           is to be found.

        dfdp : list
           a list with derivatives. There must be as many elements in
           this list as there are parameters in the model. Each element
           must be a NumPy array with the same length as *x*.

        confprob : float
           confidence probability, e.g. 0.95 (=95%).
           From this number the confidence level is derived, e.g. 0.05.
           The Confidence Band is a (1-alpha)*100% band. This implies
           that for a given value of *x* the probability that
           the 'true' value of *f* falls within these limits is
           (1-alpha)*100%.

        f : func
           the model function returning the value *y = f(p,x)*.
           *p* are the best-fit parameters as found by the method :meth:`fit` and
           *x* is the given NumPy array with independent values.

        abswei : bool
           True if weights are absolute. For absolute weights the
           unscaled covariance matrix elements are used in the calculations.
           For unit weighting (i.e. unweighted) and relative
           weighting, the covariance matrix elements are scaled with
           the value of the reduced chi squared.

        Returns
        -------
           confidence : tuple
               A tuple with the following elements, each one is a Numpy array:

               * *y*:          the model values at *x*: *y = f(p,x)*;
               * *upperband*:  the upper confidence limits;
               * *lowerband*:  the lower confidence limits.

        Notes
        -----
        If parameters were fixed in the fit, the corresponding
        error is 0 and there is no contribution to the confidence
        interval.

        """
        from scipy.stats import t

        # Given the confidence probability
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1 - confprob
        prb = 1.0 - alpha / 2
        tval = t.ppf(prb, self.dof)

        C = self.covar
        n = len(self.params)  # Number of parameters from covariance matrix
        p = self.params
        N = len(x)
        if abswei:
            covscale = 1.0
        else:
            covscale = self.rchi2_min
        df2 = np.zeros(N)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j] * dfdp[k] * C[j, k]
        df = np.sqrt(self.rchi2_min * df2)
        y = f(p, x)
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband
