# -*- coding: utf-8 -*-
r'''Tools Module
'''
from .constants import BOLTZMANN_IN_MEV_K, JOULES_TO_MEV
from multiprocessing import cpu_count, Pool  # pylint: disable=no-name-in-module
import numbers
import numpy as np
import re
from scipy import constants


def _call_bin_parallel(arg, **kwarg):
    r'''Wrapper function to work around pickling problem in Python 2.7
    '''
    return Data._bin_parallel(*arg, **kwarg)  # pylint: disable=protected-access


class Data(object):
    u'''Data class for handling multi-dimensional scattering data. If input
    file type is not supported, data can be entered manually.

    Parameters
    ----------
    Q : array_like, optional
        Default: None. **Q** in a column oriented array of [*q*\ :sub:`x`,
        *q*\ :sub:`y`, *q*\ :sub:`z`, *ℏω*, *T*]

    h : ndarray or float, optional
        Default: 0. Array of Q\ :sub:`x` in reciprocal lattice units.

    k : ndarray or float, optional
        Default: 0. Array of Q\ :sub:`y` in reciprocal lattice units.

    l : ndarray or float, optional
        Default: 0. Array of Q\ :sub:`x` in reciprocal lattice units.

    e : ndarray or float, optional
        Default: 0. Array of ℏω in meV.

    temp : ndarray or float, optional
        Default: 0. Array of sample temperatures in K.

    detector : ndarray or float, optional
        Default: 0. Array of measured counts on detector.

    monitor : ndarray or float, optional
        Default: 0. Array of measured counts on monitor.

    time : ndarray or float, optional
        Default: 0. Array of time per point in minutes.

    time_norm : bool, optional
        Default: False. If True, calls to :attr:`intensity` and :attr:`error`
        with normalize to time instead of monitor

    Attributes
    ----------
    h
    k
    l
    e
    temp
    intensity
    error
    detailed_balance_factor

    Methods
    -------
    bin
    integrate
    position
    width
    load_file
    plot

    '''
    def __init__(self, Q=None, h=0., k=0., l=0., e=0., temp=0.,
                 detector=0., monitor=0., time=0., time_norm=False, **kwargs):
        if Q is None:
            try:
                n_dim = max([len(item) for item in
                             (h, k, l, e, temp, detector, monitor, time)
                             if not isinstance(item, numbers.Number)])
            except ValueError:
                n_dim = 1

            self.Q = np.empty((n_dim, 5))

            for arg, key in zip((h, k, l, e, temp),
                                ('h', 'k', 'l', 'e', 'temp')):
                if isinstance(arg, numbers.Number):
                    arg = np.array([arg] * n_dim)
                try:
                    setattr(self, key, np.array(arg))
                except ValueError:
                    raise
        else:
            self.Q = Q

        for arg, key in zip((detector, monitor, time),
                            ('detector', 'monitor', 'time')):
            if isinstance(arg, numbers.Number):
                arg = np.array([arg] * n_dim)
            setattr(self, key, np.array(arg))

        self.m0 = np.nanmax(self.monitor)
        self.t0 = np.nanmax(self.time)

        self.time_norm = time_norm

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __add__(self, right):
        try:
            output = {'Q': right.Q, 'detector': right.detector,
                      'monitor': right.monitor, 'time': right.time}
            return self.combine_data(output, ret=True)
        except AttributeError:
            raise AttributeError('Data types cannot be combined')

    def __sub__(self, right):
        try:
            output = {'Q': right.Q, 'detector': np.negative(right.detector),
                      'monitor': right.monitor, 'time': right.time}
            return self.combine_data(output, ret=True)
        except AttributeError:
            raise AttributeError('Data types cannot be combined')

    def __mul__(self, right):
        self.detector *= right
        return self

    def __div__(self, right):
        self.detector /= right
        return self

    def __pow__(self, right):
        self.detector **= right
        return self

    @property
    def h(self):
        r'''Returns lattice parameter q\ :sub:`x`\ , *i.e.* h

        Equivalent to Q[:, 0]
        '''
        return self.Q[:, 0]

    @h.setter
    def h(self, value):
        r'''Set h to appropriate column of Q
        '''
        if isinstance(value, numbers.Number):
            value = [value] * self.Q.shape[0]
        try:
            self.Q[:, 0] = np.array(value)
        except ValueError:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.Q.shape[0]))

    @property
    def k(self):
        r'''Returns lattice parameter q\ :sub:`y`\ , *i.e.* k

        Equivalent to Q[:, 1]
        '''
        return self.Q[:, 1]

    @k.setter
    def k(self, value):
        r'''Set k to appropriate column of Q
        '''
        if isinstance(value, numbers.Number):
            value = [value] * self.Q.shape[0]
        try:
            self.Q[:, 1] = np.array(value)
        except ValueError:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.Q.shape[0]))

    @property
    def l(self):
        r'''Returns lattice parameter q\ :sub:`z`\ , *i.e.* l

        Equivalent to Q[:, 2]
        '''
        return self.Q[:, 2]

    @l.setter
    def l(self, value):
        r'''Set l to appropriate column of Q
        '''
        if isinstance(value, numbers.Number):
            value = [value] * self.Q.shape[0]
        try:
            self.Q[:, 2] = np.array(value)
        except ValueError:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.Q.shape[0]))

    @property
    def e(self):
        r'''Returns energy transfer

        Equivalent to Q[:, 3]
        '''
        return self.Q[:, 3]

    @e.setter
    def e(self, value):
        r'''Set e to appropriate column of Q
        '''
        if isinstance(value, numbers.Number):
            value = [value] * self.Q.shape[0]
        try:
            self.Q[:, 3] = np.array(value)
        except ValueError:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.Q.shape[0]))

    @property
    def temp(self):
        r'''Returns temperature

        Equivalent to Q[:, 4]
        '''
        return self.Q[:, 4]

    @temp.setter
    def temp(self, value):
        r'''Set temp to appropriate column of Q
        '''
        if isinstance(value, numbers.Number):
            value = [value] * self.Q.shape[0]
        try:
            self.Q[:, 4] = np.array(value)
        except ValueError:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.Q.shape[0]))

    @property
    def intensity(self):
        r'''Returns the monitor or time normalized intensity

        '''

        if self.time_norm:
            if self.t0 == 0:
                self.t0 = np.nanmax(self.time)
            return self.detector / self.time * self.t0
        else:
            if self.m0 == 0:
                self.m0 = np.nanmax(self.monitor)
            return self.detector / self.monitor * self.m0

    @property
    def error(self):
        r'''Returns square-root error of monitor or time normalized intensity

        '''
        if self.time_norm:
            if self.t0 == 0:
                self.t0 = np.nanmax(self.time)
            return np.sqrt(self.detector) / self.time * self.t0
        else:
            if self.m0 == 0:
                self.m0 = np.nanmax(self.monitor)
            return np.sqrt(self.detector) / self.monitor * self.m0

    @property
    def detailed_balance_factor(self):
        r'''Returns the detailed balance factor (sometimes called the Bose
        factor)

        Parameters
        ----------
            None

        Returns
        -------
        dbf : ndarray
            The detailed balance factor (temperature correction)

        '''

        return 1. - np.exp(-np.abs(self.Q[:, 3]) / BOLTZMANN_IN_MEV_K / self.temp)

    def combine_data(self, *args, **kwargs):
        r'''Combines multiple data sets

        Parameters
        ----------
        args : dictionary of ndarrays
            A dictionary (or multiple) of the data that will be added to the
            current data, with keys:
                * Q : ndarray : [h, k, l, e] with shape (N, 4,)
                * monitor : ndarray : shape (N,)
                * detector : ndarray : shape (N,)
                * temps : ndarray : shape (N,)

        Returns
        -------
        None

        '''
        Q = self.Q.copy()
        detector = self.detector.copy()  # pylint: disable=access-member-before-definition
        monitor = self.monitor.copy()  # pylint: disable=access-member-before-definition
        time = self.time.copy()  # pylint: disable=access-member-before-definition

        tols = np.array([5.e-4, 5.e-4, 5.e-4, 5.e-4, 5.e-4])

        try:
            if kwargs['tols'] is not None:
                tols = np.array(kwargs['tols'])
        except KeyError:
            pass

        for arg in args:
            combine = []
            for i in range(arg['Q'].shape[0]):
                for j in range(self.Q.shape[0]):
                    if np.all(np.abs(self.Q[j, :] - arg['Q'][i, :]) <= tols):
                        combine.append([i, j])

            for item in combine:
                monitor[item[1]] += arg['monitor'][item[0]]
                detector[item[1]] += arg['detector'][item[0]]
                time[item[1]] += arg['time'][item[0]]

            if len(combine) > 0:
                for key in ['Q', 'monitor', 'detector', 'time']:
                    arg[key] = np.delete(arg[key], (np.array(combine)[:, 0],), 0)

            Q = np.concatenate((Q, arg['Q']))
            detector = np.concatenate((detector, arg['detector']))
            monitor = np.concatenate((monitor, arg['monitor']))
            time = np.concatenate((time, arg['time']))

        order = np.lexsort([Q[:, i] for i in reversed(range(Q.shape[1]))])

        if 'ret' in kwargs and kwargs['ret']:
            return Data(Q=Q[order], monitor=monitor[order],
                        detector=detector[order], time=time[order])

        else:
            self.Q = Q[order]
            self.detector = detector[order]
            self.monitor = monitor[order]
            self.time = time[order]

    def estimate_background(self, bg_params):
        r'''Estimate the background according to ``type`` specified.

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
        '''
        if bg_params['type'] == 'constant':
            return bg_params['value']

        elif bg_params['type'] == 'percent':
            with self.intensity[self.intensity >= 0.] as inten:
                Npts = inten.size * (bg_params['value'] / 100.)
                min_vals = inten[np.argsort(inten)[:Npts]]
                background = np.average(min_vals)
                return background

        elif bg_params['type'] == 'minimum':
            return min(self.intensity)

        else:
            return 0

    def _bin_parallel(self, Q_chunk):
        r'''Performs binning by finding data chunks to bin together.
        Private function for performing binning in parallel using
        multiprocessing library

        Parameters
        ----------
        Q_chunk : ndarray
            Chunk of Q over which the binning will be performed

        Returns
        -------
        (monitor, detector, temps) : tup of ndarrays
            New monitor, detector, and temps of the binned data

        '''
        monitor = np.empty(Q_chunk.shape[0])
        detector = np.empty(Q_chunk.shape[0])
        time = np.empty(Q_chunk.shape[0])

        for i, _Q_chunk in enumerate(Q_chunk):
            _Q = self.Q
            _mon = self.monitor
            _det = self.detector
            _tim = self.time

            for j in range(_Q.shape[1]):
                _order = np.lexsort([_Q[:, j - n] for n
                                     in reversed(range(_Q.shape[1]))])
                _Q = _Q[_order]
                _mon = _mon[_order]
                _det = _det[_order]
                _tim = _tim[_order]

                chunk0 = np.searchsorted(_Q[:, j],
                                         _Q_chunk[j] - self._qstep[j] / 2.,
                                         side='left')
                chunk1 = np.searchsorted(_Q[:, j],
                                         _Q_chunk[j] + self._qstep[j] / 2.,
                                         side='right')

                if chunk0 < chunk1:
                    _Q = _Q[chunk0:chunk1, :]
                    _mon = _mon[chunk0:chunk1]
                    _det = _det[chunk0:chunk1]
                    _tim = _tim[chunk0:chunk1]

            monitor[i] = np.average(_mon[chunk0:chunk1])
            detector[i] = np.average(_det[chunk0:chunk1])
            time[i] = np.average(_tim[chunk0:chunk1])

        return (monitor, detector, time)

    def bin(self, to_bin):  # pylint: disable=unused-argument
        u'''Rebin the data into the specified shape.

        Parameters
        ----------
        to_bin : dict
            h : array_like
                Q\ :sub:`x`: [lower bound, upper bound, number of points]

            k : array_like
                Q\ :sub:`y` [lower bound, upper bound, number of points]

            l : array_like
                Q\ :sub:`z` [lower bound, upper bound, number of points]

            e : array_like
                ℏω: [lower bound, upper bound, number of points]

            temp : array_like
                *T*: [lower bound, upper bound, number of points]

        Returns
        -------
        binned_data : :class:`.Data` object
            The resulting data object with values binned to the specified bounds

        '''
        args = (to_bin[item] for item in ['h', 'k', 'l', 'e', 'temp'])
        q, qstep = (), ()
        for arg in args:
            if arg[2] == 1:
                _q, _qstep = (np.array([np.average(arg[:2])]), (arg[1] - arg[0]))
            else:
                _q, _qstep = np.linspace(arg[0], arg[1], arg[2], retstep=True)
            q += _q,
            qstep += _qstep,

        self._qstep = qstep

        Q = np.meshgrid(*q)
        Q = np.vstack((item.flatten() for item in Q)).T

        nprocs = cpu_count()  # @UndefinedVariable
        Q_chunks = [Q[n * Q.shape[0] // nprocs:(n + 1) * Q.shape[0] // nprocs] for n in range(nprocs)]
        pool = Pool(processes=nprocs)  # pylint: disable=not-callable
        outputs = pool.map(_call_bin_parallel, zip([self] * len(Q_chunks), Q_chunks))

        monitor, detector, time = (np.concatenate(arg) for arg in zip(*outputs))

        return Data(Q=Q, monitor=monitor, detector=detector, time=time, m0=self.m0, t0=self.t0)

    def integrate(self, background=None, **kwargs):
        r'''Returns the integrated intensity within given bounds

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

        '''
        if background is not None:
            background = self.estimate_background(kwargs['background'])
        else:
            background = 0

        result = 0
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for i in range(4):
                result += np.trapz(self.intensity[to_fit] - background, x=self.Q[to_fit, i])
        else:
            for i in range(4):
                result += np.trapz(self.intensity - background, x=self.Q[:, i])

        return result

    def position(self, background=None, **kwargs):
        r'''Returns the position of a peak within the given bounds

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

        '''
        if background is not None:
            background = self.estimate_background(kwargs['background'])
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
        r'''Returns the mean-squared width of a peak within the given bounds

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

        '''
        if background is not None:
            background = self.estimate_background(kwargs['background'])
        else:
            background = 0

        result = ()
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = (self.Q[to_fit, j] - self.position(**kwargs)[j]) ** 2 * (self.intensity[to_fit] - background)
                    _result += np.trapz(y, x=self.Q[to_fit, i]) / self.integrate(**kwargs)
                result += (_result,)
        else:
            for j in range(4):
                _result = 0
                for i in range(4):
                    y = (self.Q[:, j] - self.position(**kwargs)[j]) ** 2 * (self.intensity - background)
                    _result += np.trapz(y, x=self.Q[:, i]) / self.integrate(**kwargs)
                result += (_result,)

        return result

    def plot(self, x, y, z=None, w=None, show_err=True, to_bin=None,
             plot_options=None, fit_options=None, smooth_options=None,
             output_file='', show_plot=True, **kwargs):
        r'''Plots the data in the class. x and y must at least be specified,
        and z and/or w being specified will produce higher dimensional plots
        (contour and volume, respectively).

        Parameters
        ----------
        x : str
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        y : str
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        z : str, optional
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        w : str, optional
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each 'h', 'k', 'l', 'e',
            and 'temp'

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: False

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported.

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`

        Returns
        -------
        None

        '''
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors  # @UnusedImport
        except ImportError:
            ImportError('Matplotlib >= 1.3.0 is necessary for plotting.')

        if to_bin is None:
            to_bin = {}
        if plot_options is None:
            plot_options = {}
        if fit_options is None:
            fit_options = {}
        if smooth_options is None:
            smooth_options = {'sigma': 0}

        args = {'x': x, 'y': y, 'z': z, 'w': w}
        options = ['h', 'k', 'l', 'e', 'temp', 'intensity']

        in_axes = np.array([''] * len(options))
        for key, value in args.items():
            if value is not None:
                in_axes[np.where(np.array(options) == value[0])] = key

        if to_bin:
            binned_data = self.bin(to_bin)
            to_plot = np.where(binned_data.monitor > 0)
            dims = {'h': binned_data.h[to_plot],
                    'k': binned_data.k[to_plot],
                    'l': binned_data.l[to_plot],
                    'e': binned_data.e[to_plot],
                    'temp': binned_data.temp[to_plot],
                    'intensity': binned_data.intensity[to_plot],
                    'error': binned_data.error[to_plot]}
        else:
            to_plot = np.where(self.monitor > 0)
            dims = {'h': self.h[to_plot],
                    'k': self.k[to_plot],
                    'l': self.l[to_plot],
                    'e': self.e[to_plot],
                    'temp': self.temp[to_plot],
                    'intensity': self.intensity[to_plot],
                    'error': self.error[to_plot]}

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            dims['intensity'] = gaussian_filter(dims['intensity'],
                                                **smooth_options)

        x = dims[args['x']]
        y = dims[args['y']]

        if z is not None and w is not None:
            try:
                z = dims[args['z']]
                w = dims[args['w']]

                x, y, z, w = (np.ma.masked_where(w <= 0, x),
                              np.ma.masked_where(w <= 0, y),
                              np.ma.masked_where(w <= 0, z),
                              np.ma.masked_where(w <= 0, w))

                from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

                fig = plt.figure()
                axis = fig.add_subplot(111, projection='3d')

                axis.scatter(x, y, z, c=w, linewidths=0, vmin=1.e-4,
                             vmax=0.1, norm=colors.LogNorm())

            except KeyError:
                raise

        elif z is not None and w is None:
            try:
                z = dims[args['z']]

#                 x, y, z = (np.ma.masked_where(z <= 0, x),
#                            np.ma.masked_where(z <= 0, y),
#                            np.ma.masked_where(z <= 0, z))
                X, Y = np.meshgrid(np.linspace(x.min(), x.max(), np.around(np.abs(np.unique(x)-np.roll(np.unique(x), 1))[1], decimals=4)),
                                   np.linspace(y.min(), y.max(), np.around(np.abs(np.unique(y)-np.roll(np.unique(y), 1))[1], decimals=4)))
                
                from scipy.interpolate import griddata
                Z = griddata((x, y), z, (X, Y))
                
                plt.pcolormesh(X, Y, Z, **plot_options)
            except KeyError:
                raise
        else:
            if not plot_options:
                plot_options['fmt'] = 'rs'

            if show_err:
                err = np.sqrt(dims['intensity'])
                plt.errorbar(x, y, yerr=err, **plot_options)
            else:
                plt.errorbar(x, y, **plot_options)

            if fit_options:
                try:
                    from .kmpfit import Fitter
                except ImportError:
                    raise

                def residuals(params, data):
                    funct, x, y, err = data

                    return (y - funct(params, x)) / err

                fitobj = Fitter(residuals, data=(fit_options['function'], x, y,
                                                 np.sqrt(dims['intensity'])))
                if 'fixp' in fit_options:
                    fitobj.parinfo = [{'fixed': fix} for fix in
                                      fit_options['fixp']]

                try:
                    fitobj.fit(params0=fit_options['p'])
                    fit_x = np.linspace(min(x), max(x), len(x) * 10)
                    fit_y = fit_options['function'](fitobj.params, fit_x)
                    plt.plot(fit_x, fit_y, '{0}-'.format(plot_options['fmt'][0]))

                    param_string = u'\n'.join(['p$_{{{0:d}}}$: {1:.3f}'.format(i, p)
                                               for i, p in enumerate(fitobj.params)])
                    chi2_params = u'$\chi^2$: {0:.3f}\n\n'.format(fitobj.chi2_min) + param_string

                    plt.annotate(chi2_params, xy=(0.05, 0.95), xycoords='axes fraction',
                                 horizontalalignment='left', verticalalignment='top',
                                 bbox=dict(alpha=0.75, facecolor='white', edgecolor='none'))

                except Exception as mes:  # pylint: disable=broad-except
                    print("Something wrong with fit: {0}".format(mes))

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass


def load(files, filetype='auto', tols=1e-4):
    r'''Loads one or more files and creates a :class:`Data` object with the
    loaded data.

    Parameters
    ----------
    files : str or tuple of str
        A file or non-keyworded list of files containing data for input.

    filetype : str, optional
        Default: `'auto'`. Specify file type; Currently supported file types
        are SPICE, ICE, and ICP. By default, the function will attempt to
        determine the filetype automatically.

    tols : float or array_like
        Default: `1e-4`. A float or array of shape `(5,)` giving tolerances for
        combining multiple files. If multiple points are within the given
        tolerances then they will be combined into a single point. If a float
        is given, tolerances will all be the same for all variables in **Q**.
        If an array is given tolerances should be in the format
        `[h, k, l, e, temp]`.

    Returns
    -------
    Data : object
        A :class:`Data` object populated with the data from the input file or
        files.

    '''
    if isinstance(files, str):
        files = (files,)

    if isinstance(tols, numbers.Number):
        tols = [tols for i in range(5)]

    for filename in files:
        if filetype == 'auto':
            try:
                filetype = detect_filetype(filename)
            except ValueError:
                raise

        if filetype == 'SPICE':
            data_keys = {'monitor': 'monitor', 'detector': 'detector',
                         'time': 'time'}
            Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'tvti'}
            raw_data = {}

            with open(filename) as f:
                for line in f:
                    if 'col_headers' in line:
                        args = next(f).split()
                        headers = [head for head in args[1:]]

            args = np.genfromtxt(filename, unpack=True, dtype=np.float64)

            _t0 = 60.

        elif filetype == 'ICE':
            data_keys = {'detector': 'Detector', 'monitor': 'Monitor',
                         'time': 'Time'}
            Q_keys = {'h': 'QX', 'k': 'QY', 'l': 'QZ', 'e': 'E',
                      'temp': 'Temp'}
            raw_data = {}
            _t0 = 60.

            with open(filename) as f:
                for line in f:
                    if 'Columns' in line:
                        args = line.split()
                        headers = [head for head in args[1:]]

            args = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                                 unpack=True, comments="#", dtype=np.float64)

        elif filetype == 'ICP':
            data_keys = {'detector': 'Counts', 'time': 'min'}
            Q_keys = {'h': 'Q(x)', 'k': 'Q(y)', 'l': 'Q(z)', 'e': 'E',
                      'temp': 'T-act'}
            raw_data = {}
            _t0 = 1.

            with open(filename) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        _length = int(re.findall(r"(?='(.*?)')", line)[-2])
                        [_m0, _prf] = [float(i) for i in re.findall(r"(?='(.*?)')", line)[-4].split()]
                    if 'Q(x)' in line:
                        args = line.split()
                        headers = [head for head in args]

            args = np.genfromtxt(filename, unpack=True, dtype=np.float64, skip_header=12)

            raw_data['monitor'] = np.empty(args[0].shape)
            raw_data['monitor'].fill(_m0 * _prf)

        else:
            raise ValueError('Filetype not supported.')

        for key, value in data_keys.items():
            try:
                raw_data[key] = args[headers.index(value)]
            except ValueError:
                print("ValueError: '{0}' is not in list.".format(value))
                raw_data[key] = np.ones(args[0].shape)

        _Q_dict = {}
        for key, value in Q_keys.items():
            try:
                _Q_dict[key] = args[headers.index(value)]
            except ValueError:
                print("ValueError: '{0}' is not in list.".format(value))
                _Q_dict[key] = np.ones(args[0].shape)

        raw_data['time'] /= _t0
        raw_data['Q'] = build_Q(_Q_dict)

        del _Q_dict, args

        try:
            _data_object.combine_data(raw_data)
        except NameError:
            _data_object = Data(**raw_data)

    return _data_object


def save(obj, filename, format='ascii', **kwargs):
    '''Saves a given object to a file in a specified format.
    
    Parameters
    ----------
    obj : :class:`Data`
        A :class:`Data` object to be saved to disk
    
    filename : str
        Path to file where data will be saved
    
    format : str
        Default: `'ascii'`. Data can either be saved in `'ascii'`,
        human-readable format, `'binary'` format, or `'nexus'` format.
    '''
    output = np.hstack((obj.Q, obj.detector, obj.monitor, obj.time))
    
    if format == 'ascii':
        np.savetxt(filename, output, **kwargs)
    elif format == 'binary':
        pass
    elif format == 'nexus':
        pass
    else:
        raise ValueError("""Format not supported. Please use 'ascii', \
                            'binary', 'pickle' or 'nexus'""")


def build_Q(vars, **kwargs):
    u'''Method for constructing **Q**\ (*q*, ℏω, temp) from h, k, l,
    energy, and temperature

    Parameters
    ----------
    vars : dict
        A dictionary of the `h`, `k`, `l`, `e` and `temp` arrays to form into
        a column oriented array

    Returns
    -------
    Q : ndarray
        Returns **Q**\ (h, k, l, e, temp) with shape (N, 5) in a column oriented
        array.

    '''
    return np.vstack((vars[i].flatten() for i in
                      ['h', 'k', 'l', 'e', 'temp'])).T


def detect_filetype(file):
    u'''Simple method for quickly determining filetype of a given input file.
    
    Parameters
    ----------
    file : str
        File path
    
    Returns
    -------
    filetype : str
        The filetype of the given input file
    '''
    if file[-3:] == 'nxs':
        return 'nexus'
    elif file[-4:] == 'iexy':
        return 'iexy'
    else:
        with open(file) as f:
            first_line = f.readline()
            second_line = f.readline()
            if '#ICE' in first_line:
                return 'ICE'
            elif '# scan' in first_line:
                return 'SPICE'
            elif 'Filename' in second_line:
                return 'ICP'
            else:
                raise ValueError('Unknown filetype.')


class Energy():
    u'''Class containing the most commonly used properties of a neutron beam
    given some initial input, e.g. energy, wavelength, velocity, wavevector,
    temperature, or frequency. At least one input must be supplied.

    Parameters
    ----------
    energy : float
        Neutron energy in millielectron volts (meV)
    wavelength : float
        Neutron wavelength in angstroms (Å)
    velocity : float
        Neutron velocity in meters per second (m/s)
    wavevector : float
        Neutron wavevector k in inverse angstroms (1/Å)
    temperature : float
        Neutron temperature in kelvin (K)
    frequency : float
        Neutron frequency in terahertz (THz)

    Returns
    -------
    Energy object
        The energy object containing the properties of the neutron beam
    '''
    def __init__(self, energy=None, wavelength=None, velocity=None,
                 wavevector=None, temperature=None, frequency=None):
        
        self._update_values(energy, wavelength, velocity,
                 wavevector, temperature, frequency)

    def _update_values(self, energy=None, wavelength=None, velocity=None,
                 wavevector=None, temperature=None, frequency=None):
        try:
            if energy is None:
                if wavelength is not None:
                    self.en = constants.h ** 2 / (2. * constants.m_n * (wavelength / 1.e10) ** 2) * JOULES_TO_MEV
                elif velocity is not None:
                    self.en = 1. / 2. * constants.m_n * velocity ** 2 * JOULES_TO_MEV
                elif wavevector is not None:
                    self.en = (constants.h ** 2 / (2. * constants.m_n * ((2. * np.pi / wavevector) / 1.e10) ** 2) * JOULES_TO_MEV)
                elif temperature is not None:
                    self.en = constants.k * temperature * JOULES_TO_MEV
                elif frequency is not None:
                    self.en = (constants.hbar * frequency * 2. * np.pi * JOULES_TO_MEV * 1.e12)
            else:
                self.en = energy

            self.wavelen = np.sqrt(constants.h ** 2 / (2. * constants.m_n * self.energy / JOULES_TO_MEV)) * 1.e10
            self.vel = np.sqrt(2. * self.energy / JOULES_TO_MEV / constants.m_n)
            self.wavevec = 2. * np.pi / self.wavelength
            self.temp = self.energy / constants.k / JOULES_TO_MEV
            self.freq = (self.energy / JOULES_TO_MEV / constants.hbar / 2. / np.pi / 1.e12)

        except AttributeError:
            raise AttributeError('''You must define at least one of the \
                                    following: energy, wavelength, velocity, \
                                    wavevector, temperature, frequency''')
    
    @property
    def energy(self):
        return self.en
    
    @energy.setter
    def energy(self, value):
        self._update_values(energy=value)
    
    @property
    def wavelength(self):
        return self.wavelen

    @wavelength.setter
    def wavelength(self, value):
        self._update_values(wavelength=value)
    
    @property
    def wavevector(self):
        return self.wavevec
    
    @wavevector.setter
    def wavevector(self, value):
        self._update_values(wavevector=value)
    
    @property
    def temperature(self):
        return self.temp

    @temperature.setter
    def temperature(self, value):
        self._update_values(temperature=value)
    
    @property
    def frequency(self):
        return self.freq

    @frequency.setter
    def frequency(self, value):
        self._update_values(frequency=value)
    
    @property
    def velocity(self):
        return self.vel
    
    @velocity.setter
    def velocity(self, value):
        self._update_values(velocity=value)


    @property
    def values(self):
        '''Prints all of the properties of the Neutron beam

        Parameters
        ----------
        None

        Returns
        -------
        values : string
            A string containing all the properties of the neutron including
            respective units
        '''
        return u'''
Energy: {0:3.3f} meV
Wavelength: {1:3.3f} Å
Wavevector: {2:3.3f} 1/Å
Velocity: {3:3.3f} m/s
Temperature: {4:3.3f} K
Frequency: {5:3.3f} THz
'''.format(self.energy, self.wavelength, self.wavevector, self.velocity,
           self.temperature, self.frequency)
