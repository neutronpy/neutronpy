# -*- coding: utf-8 -*-
r'''Tools Module
'''
from .constants import BOLTZMANN_IN_MEV_K, JOULES_TO_MEV
from multiprocessing import cpu_count, Pool  # pylint: disable=no-name-in-module
import numpy as np
import re
from scipy import constants


def _call_bin_parallel(arg, **kwarg):
    r'''Wrapper function to work around pickling problem in Python 2.7
    '''
    return Data.bin_parallel(*arg, **kwarg)


class Data(object):
    r'''Data class for handling multi-dimensional TAS data. If input file type
    is not supported, data can be entered manually.

    Parameters
    ----------
    h : ndarray, optional
        Array of :math:`Q_x` in reciprocal lattice units.

    k : ndarray, optional
        Array of :math:`Q_y` in reciprocal lattice units.

    l : ndarray, optional
        Array of :math:`Q_z` in reciprocal lattice units.

    e : ndarray, optional
        Array of :math:`\hbar \omega` in meV.

    detector : ndarray, optional
        Array of measured counts on detector.

    monitor : ndarray, optional
        Array of measured counts on monitor.

    temp : ndarray, optional
        Array of sample temperatures in K.

    Returns
    -------
    Data Class
        The data class for handling Triple Axis Spectrometer Data

    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __add__(self, right):
        try:
            output = {'Q': right.Q, 'temp': right.temp, 'detector': right.detector, 'monitor': right.monitor}
            return self.combine_data(output, ret=True)
        except AttributeError:
            raise AttributeError('Data types cannot be combined')

    def __sub__(self, right):
        try:
            output = {'Q': right.Q, 'temp': right.temp, 'detector': np.negative(right.detector), 'monitor': right.monitor}
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

    def load_file(self, *files, **kwargs):
        r'''Loads one or more files in either SPICE, ICE or ICP formats

        Parameters
        ----------
        files : string
            A file or non-keyworded list of files containing data for input.

        mode : string
            Specify file type (SPICE | ICE | ICP). Currently only file types
            supported.

        Returns
        -------
        None

        '''
        if kwargs['mode'] == 'SPICE':
            keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'monitor': 'monitor', 'detector': 'detector', 'temp': 'temp'}
            for filename in files:
                output = {}
                with open(filename) as f:
                    for line in f:
                        if 'col_headers' in line:
                            args = next(f).split()
                            headers = [head.replace('.', '') for head in args[1:]]

                args = np.loadtxt(filename, unpack=True, dtype=np.float64)

                for key, value in keys.items():
                    output[key] = args[headers.index(value)]

                if not hasattr(self, 'Q'):
                    for key, value in output.items():
                        setattr(self, key, value)
                    self.Q = self._build_Q(**kwargs)
                else:
                    output['Q'] = self._build_Q(output=output, **kwargs)
                    self.combine_data(output)

        if kwargs['mode'] == 'ICE':
            keys = {'h': 'QX', 'k': 'QY', 'l': 'QZ', 'e': 'E', 'detector': 'Detector', 'monitor': 'Monitor', 'temp': 'Temp'}
            for filename in files:
                output = {}
                with open(filename) as f:
                    for line in f:
                        if 'Columns' in line:
                            args = line.split()
                            headers = [head.replace('(', '').replace(')', '').replace('-', '') for head in args[1:]]

                args = np.genfromtxt(filename, comments="#", dtype=np.float64, unpack=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))

                for key, value in keys.items():
                    output[key] = args[headers.index(value)]

                if not hasattr(self, 'Q'):
                    for key, value in output.items():
                        setattr(self, key, value)
                    self.Q = self._build_Q(**kwargs)
                else:
                    output['Q'] = self._build_Q(output=output, **kwargs)
                    self.combine_data(output)

        if kwargs['mode'] == 'ICP':
            keys = {'h': 'Qx', 'k': 'Qy', 'l': 'Qz', 'e': 'E', 'detector': 'Counts', 'temp': 'Tact'}
            for filename in files:
                output = {}
                with open(filename) as f:
                    for i, line in enumerate(f):
                        if i == 0:
                            self.length = int(re.findall(r"(?='(.*?)')", line)[-2])
                            self.m0 = float(re.findall(r"(?='(.*?)')", line)[-4].split()[0])
                        if 'Q(x)' in line:
                            args = line.split()
                            headers = [head.replace('(', '').replace(')', '').replace('-', '') for head in args]
                args = np.loadtxt(filename, unpack=True, dtype=np.float64, skiprows=12)

                for key, value in keys.items():
                    output[key] = args[headers.index(value)]

                output['monitor'] = np.zeros(output['detector'].shape) + self.m0

                if not hasattr(self, 'Q'):
                    for key, value in output.items():
                        setattr(self, key, value)
                    self.Q = self._build_Q(**kwargs)
                else:
                    output['Q'] = self._build_Q(output=output, **kwargs)
                    self.combine_data(output)

    def _build_Q(self, **kwargs):
        r'''Internal method for constructing :math:`Q(q, hw)` from h, k, l,
        and energy

        Parameters
        ----------
        output : dictionary, optional
            A dictionary of the h, k, l, and e arrays to form into a column
            oriented array

        Returns
        -------
        Q : ndarray, shape (N, 4,)
            Returns Q (h, k, l, e) in a column oriented array.

        '''
        args = ()
        if 'output' in kwargs:
            for i in ['h', 'k', 'l', 'e']:
                args += (kwargs['output'][i],)
        else:
            for i in ['h', 'k', 'l', 'e']:
                args += (getattr(self, i),)

        return np.vstack((item.flatten() for item in args)).T

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
        monitor, detector, Q, temp = self.monitor.copy(), self.detector.copy(), self.Q.copy(), self.temp.copy()  # pylint: disable=access-member-before-definition

        for arg in args:
            combine = []
            for i in range(arg['Q'].shape[0]):
                for j in range(self.Q.shape[0]):
                    if np.all(self.Q[j, :] == arg['Q'][i, :]):
                        combine.append([i, j])

            for item in combine:
                monitor[item[1]] += arg['monitor'][item[0]]
                detector[item[1]] += arg['detector'][item[0]]

            if len(combine) > 0:
                for key in ['Q', 'monitor', 'detector', 'temp']:
                    arg[key] = np.delete(arg[key], (np.array(combine)[:, 0],), 0)

            Q = np.concatenate((Q, arg['Q']))
            detector = np.concatenate((detector, arg['detector']))
            monitor = np.concatenate((monitor, arg['monitor']))
            temp = np.concatenate((temp, arg['temp']))

            order = np.lexsort((Q[:, 3], Q[:, 2], Q[:, 1], Q[:, 0]))

        if 'ret' in kwargs and kwargs['ret']:
            new = Data(Q=Q[order], temp=temp[order], monitor=monitor[order], detector=detector[order])

            for i, var in enumerate(['h', 'k', 'l', 'e']):
                setattr(new, var, new.Q[:, i])

            return new

        else:
            self.Q = Q[order]
            self.monitor = monitor[order]
            self.detector = detector[order]
            self.temp = temp[order]

            for i, var in enumerate(['h', 'k', 'l', 'e']):
                setattr(self, var, self.Q[:, i])

    def intensity(self, **kwargs):
        r'''Returns the monitor normalized intensity

        Parameters
        ----------
        m0 : float, optional
            Desired monitor to normalize the intensity. If not specified, m0
            is set to the max monitor.

        Returns
        -------
        intensity : ndarray
            The monitor normalized intensity scaled by m0

        '''
        try:
            m0 = kwargs['m0']
        except KeyError:
            try:
                m0 = self.m0
            except AttributeError:
                self.m0 = m0 = np.nanmax(self.monitor)

        return self.detector / self.monitor * m0

    def error(self, **kwargs):
        r'''Returns square-root error of monitor normalized intensity

        Parameters
        ----------
        m0 : float, optional
            Desired monitor to normalize the intensity

        Returns
        -------
        error : ndarray
            The square-root error of the monitor normalized intensity

        '''
        return np.sqrt(np.abs(self.intensity(**kwargs)))

    def detailed_balance_factor(self, **kwargs):
        r'''Returns the detailed balance factor (sometimes called the Bose
        factor)

        Parameters
        ----------
        temp : float, optional
            If not already a property of the class, the sample temperature
            can be specified as a float.

        Returns
        -------
        dbf : ndarray
            The detailed balance factor (temperature correction)

        '''
        try:
            self.temps = np.zeros(self.Q.shape[0]) + kwargs['temp']
        except KeyError:
            pass

        return np.exp(-self.Q[3] / BOLTZMANN_IN_MEV_K / self.temps)

    def bg_estimate(self, perc):
        r'''Estimate the background by averaging the
        bottom perc % of points that are >= 0
        '''
        inten = self.intensity()[self.intensity() >= 0.]
        Npts = inten.size * (perc / 100.)
        min_vals = inten[np.argsort(inten)[:Npts]]
        bg = np.average(min_vals)
        return bg

    def bin_parallel(self, Q_chunk):
        r'''Performs binning by finding data chunks to bin together.
        Private function for performing binning in parallel using
        multiprocessing library

        Parameters
        ----------
        Q_chunk : ndarray
            Chunk of Q over which the binning will be performed

        Returns
        -------
        (monitor, detector, temps) : tuple of ndarrays
            New monitor, detector, and temps of the binned data

        '''
        monitor, detector, temps = np.zeros(Q_chunk.shape[0]), np.zeros(Q_chunk.shape[0]), np.zeros(Q_chunk.shape[0])

        for i in range(Q_chunk.shape[0]):
            chunk0 = np.where((self.Q[:, 0] - Q_chunk[i, 0]) ** 2 / (self._qstep[0] / 2.) ** 2 < 1.)

            if len(chunk0[0]) > 0:
                _Q, _mon, _det, _temp = self.Q[chunk0, :][0], self.monitor[chunk0], self.detector[chunk0], self.temp[chunk0]
                chunk1 = np.where((_Q[:, 1] - Q_chunk[i, 1]) ** 2 / (self._qstep[1] / 2.) ** 2 < 1.)

                if len(chunk1[0]) > 0:
                    _Q, _mon, _det, _temp = _Q[chunk1, :][0], _mon[chunk1], _det[chunk1], _temp[chunk1]
                    chunk2 = np.where((_Q[:, 2] - Q_chunk[i, 2]) ** 2 / (self._qstep[2] / 2.) ** 2 < 1.)

                    if len(chunk2[0]) > 0:
                        _Q, _mon, _det, _temp = _Q[chunk2, :][0], _mon[chunk2], _det[chunk2], _temp[chunk2]
                        chunk3 = np.where((_Q[:, 1] - Q_chunk[i, 1]) ** 2 / (self._qstep[3] / 2.) ** 2 < 1.)

                        if len(chunk3[0]) > 0:
                            _Q, _mon, _det, _temp = _Q[chunk3, :][0], _mon[chunk3], _det[chunk3], _temp[chunk3]
                            chunk4 = np.where((_temp - Q_chunk[i, 4]) ** 2 / (self._qstep[4] / 2.) ** 2 < 1.)

                            if len(chunk4[0]) > 0:
                                _Q, _mon, _det, _temp = _Q[chunk4, :][0], _mon[chunk4], _det[chunk4], _temp[chunk4]

                                monitor[i] = np.average(_mon)
                                detector[i] = np.average(_det)
                                temps[i] = np.average(_temp)

        return (monitor, detector, temps)

    def bin(self, h, k, l, e, temp):  # pylint: disable=unused-argument
        r'''Rebin the data into the specified shape.

        Parameters
        ----------
        h : list
            :math:`Q_x`: [lower bound, upper bound, number of points]

        k : list
            :math:`Q_y`: [lower bound, upper bound, number of points]

        l : list
            :math:`Q_z`: [lower bound, upper bound, number of points]

        e : list
            :math:`\hbar \omega`: [lower bound, upper bound, number of points]

        temp : list
            :math:`T`: [lower bound, upper bound, number of points]

        Returns
        -------
        (Q, monitor, detector, temp) : tuple of ndarray
            The resulting values binned to the specified bounds

        '''
        args = (h, k, l, e, temp)
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

        monitor, detector, temp = (np.concatenate(arg) for arg in zip(*outputs))

        return Q, monitor, detector, temp

    def integrate(self, **kwargs):
        r'''Returns the integrated intensity within given bounds

        Parameters
        ----------
        bounds : Boolean, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : float
            The integrated intensity either over all data, or within
            specified boundaries

        '''
        if 'bg' in kwargs:
            if kwargs['bg'][0] == 'c':
                bg = np.float(kwargs['bg'][1:])
            elif kwargs['bg'][0] == 'p':
                bg = self.bg_estimate(kwargs['bg'][1:])
        else:
            bg = 0

        result = 0
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for i in range(4):
                result += np.trapz(self.intensity()[to_fit] - bg, x=self.Q[to_fit, i])
        else:
            for i in range(4):
                result += np.trapz(self.intensity() - bg, x=self.Q[:, i])

        return result

    def position(self, **kwargs):
        r'''Returns the position of a peak within the given bounds

        Parameters
        ----------
        bounds : Boolean, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : tuple
            The result is a tuple with position in each dimension of Q,
            (h, k, l, e)

        '''

        if 'bg' in kwargs:
            if kwargs['bg'][0] == 'c':
                bg = np.float(kwargs['bg'][1:])
            elif kwargs['bg'][0] == 'p':
                bg = self.bg_estimate(kwargs['bg'][1:])
        else:
            bg = 0

        result = ()
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for j in range(4):
                _result = 0
                for i in range(4):
                    _result += np.trapz(self.Q[to_fit, j] * (self.intensity()[to_fit] - bg), x=self.Q[to_fit, i]) / self.integrate(**kwargs)
                result += (_result,)
        else:
            for j in range(4):
                _result = 0
                for i in range(4):
                    _result += np.trapz(self.Q[:, j] * (self.intensity() - bg), x=self.Q[:, i]) / self.integrate(**kwargs)
                result += (_result,)

        return result

    def width(self, **kwargs):
        r'''Returns the mean-squared width of a peak within the given bounds

        Parameters
        ----------
        bounds : Boolean, optional
            A boolean expression representing the bounds inside which the
            calculation will be performed

        Returns
        -------
        result : tuple
            The result is a tuple with the width in each dimension of Q,
            (h, k, l, e)

        '''
        if 'bg' in kwargs:
            if kwargs['bg'][0] == 'c':
                bg = np.float(kwargs['bg'][1:])
            elif kwargs['bg'][0] == 'p':
                bg = self.bg_estimate(kwargs['bg'][1:])
        else:
            bg = 0

        result = ()
        if 'bounds' in kwargs:
            to_fit = np.where(kwargs['bounds'])
            for j in range(4):
                _result = 0
                for i in range(4):
                    _result += np.trapz((self.Q[to_fit, j] - self.position(**kwargs)[j]) ** 2 * (self.intensity()[to_fit] - bg), x=self.Q[to_fit, i]) / self.integrate(**kwargs)
                result += (_result,)
        else:
            for j in range(4):
                _result = 0
                for i in range(4):
                    _result += np.trapz((self.Q[:, j] - self.position(**kwargs)[j]) ** 2 * (self.intensity() - bg), x=self.Q[:, i]) / self.integrate(**kwargs)
                result += (_result,)

        return result

    def plot(self, x, y, z=None, w=None, show_err=False, bounds=None, plot_options=None, fit_options=None,
             smooth_options=None, output_file='', show_plot=True, **kwargs):
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

        if bounds is None:
            bounds = {}
        if plot_options is None:
            plot_options = {'fmt': 'rs'}
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

        if bounds:
            Q, monitor, detector, temp = self.bin(*(bounds[opt] for opt in options[:-1]))
            to_plot = np.where(monitor > 0)
            dims = {'h': Q[to_plot, 0][0], 'k': Q[to_plot, 1][0], 'l': Q[to_plot, 2][0], 'e': Q[to_plot, 3][0],
                'temp': temp[to_plot], 'intensity': detector[to_plot] / monitor[to_plot] * self.m0}
        else:
            to_plot = np.where(self.monitor > 0)
            dims = {'h': self.Q[to_plot, 0][0], 'k': self.Q[to_plot, 1][0], 'l': self.Q[to_plot, 2][0], 'e': self.Q[to_plot, 3][0],
                'temp': self.temp[to_plot], 'intensity': self.detector[to_plot] / self.monitor[to_plot] * self.m0}

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            dims['intensity'] = gaussian_filter(dims['intensity'], **smooth_options)

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
                z = dims[kwargs['z']]

                x, y, z = (np.ma.masked_where(z <= 0, x),
                           np.ma.masked_where(z <= 0, y),
                           np.ma.masked_where(z <= 0, z))

                plt.pcolormesh(x, y, z, vmin=1.e-4, vmax=0.1,
                               norm=colors.LogNorm())
            except KeyError:
                pass
        else:
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

                fitobj = Fitter(residuals, data=(fit_options['function'], x, y, np.sqrt(dims['intensity'])))
                if 'fixp' in fit_options:
                    fitobj.parinfo = [{'fixed': fix} for fix in fit_options['fixp']]

                try:
                    fitobj.fit(params0=fit_options['p'])
                    fit_x = np.linspace(min(x), max(x), len(x) * 10)
                    fit_y = fit_options['function'](fitobj.params, fit_x)
                    plt.plot(fit_x, fit_y, '{0}-'.format(plot_options['fmt'][0]))

                    param_string = u'\n'.join(['p$_{{{0:d}}}$: {1:.3f}'.format(i, p) for i, p in enumerate(fitobj.params)])
                    chi2_params = u'$\chi^2$: {0:.3f}\n\n'.format(fitobj.chi2_min) + param_string

                    plt.annotate(chi2_params, xy=(0.05, 0.95), xycoords='axes fraction',
                                 horizontalalignment='left', verticalalignment='top',
                                 bbox=dict(alpha=0.75, facecolor='white', edgecolor='none'))

                except Exception as mes:
                    print("Something wrong with fit: {0}".format(mes))

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass


class Neutron():
    r'''Class containing the most commonly used properties of a neutron beam
    given some initial input, e.g. energy, wavelength, wavevector,
    temperature, or frequency'''

    def __init__(self, e=None, l=None, v=None, k=None, temp=None, freq=None):
        if e is None:
            if l is not None:
                self.e = constants.h ** 2 / (2. * constants.m_n * (l / 1.e10) ** 2) * JOULES_TO_MEV
            elif v is not None:
                self.e = 1. / 2. * constants.m_n * v ** 2 * JOULES_TO_MEV
            elif k is not None:
                self.e = (constants.h ** 2 / (2. * constants.m_n * ((2. * np.pi / k) / 1.e10) ** 2) * JOULES_TO_MEV)
            elif temp is not None:
                self.e = constants.k * temp * JOULES_TO_MEV
            elif freq is not None:
                self.e = (constants.hbar * freq * 2. * np.pi * JOULES_TO_MEV * 1.e12)
        else:
            self.e = e

        self.l = np.sqrt(constants.h ** 2 / (2. * constants.m_n * self.e / JOULES_TO_MEV)) * 1.e10
        self.v = np.sqrt(2. * self.e / JOULES_TO_MEV / constants.m_n)
        self.k = 2. * np.pi / self.l
        self.temp = self.e / constants.k / JOULES_TO_MEV
        self.freq = (self.e / JOULES_TO_MEV / constants.hbar / 2. / np.pi / 1.e12)

    def print_values(self):
        print(u'''
Energy: {0:3.3f} meV
Wavelength: {1:3.3f} Å
Wavevector: {2:3.3f} 1/Å
Velocity: {3:3.3f} m/s
Temperature: {4:3.3f} K
Frequency: {5:3.3f} THz
'''.format(self.e, self.l, self.k, self.v, self.temp, self.freq))
