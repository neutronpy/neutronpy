# -*- coding: utf-8 -*-
r'''Data handling

'''
from multiprocessing import cpu_count, Pool
import numbers
import numpy as np
from .constants import BOLTZMANN_IN_MEV_K
from .energy import Energy
from .plot import PlotData


def _call_bin_parallel(arg, **kwarg):
    r'''Wrapper function to work around pickling problem in Python 2.7
    '''
    return Data._bin_parallel(*arg, **kwarg)


class Analysis(object):
    r'''Class containing methods for the Data class

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

    '''
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

        return 1. - np.exp(-self.Q[:, 3] / BOLTZMANN_IN_MEV_K / self.temp)

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
        r'''Returns the neutron scattering function, i.e. the detector counts
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
        '''
        ki = Energy(energy=ei).wavevector
        kf = Energy(energy=ei - self.e).wavevector

        return (4 * np.pi / (material.total_scattering_cross_section) * ki /
                kf * self.detector)

    def dynamic_susceptibility(self, material, ei):
        r'''Returns the dynamic susceptibility
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
        '''
        return (self.scattering_function(material, ei) *
                self.detailed_balance_factor)

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
            inten = self.intensity[self.intensity >= 0.]
            Npts = inten.size * (bg_params['value'] / 100.)
            min_vals = inten[np.argsort(inten)[:Npts]]
            background = np.average(min_vals)
            return background

        elif bg_params['type'] == 'minimum':
            return min(self.intensity)

        else:
            return 0


class Data(PlotData, Analysis):
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
    combine_data
    subtract_background
    integrate
    position
    width
    estimate_background
    subtract_background
    dynamic_susceptibility
    scattering_function
    plot

    '''
    def __init__(self, Q=None, h=0., k=0., l=0., e=0., temp=0.,
                 detector=0., monitor=0., error=None, time=0., time_norm=False,
                 **kwargs):
        if Q is None:
            try:
                n_dim = max([len(item) for item in
                             (h, k, l, e, temp, detector, monitor, time)
                             if not isinstance(item, numbers.Number)])
            except (ValueError, UnboundLocalError):
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
            n_dim = Q.shape[1]

        for arg, key in zip((detector, monitor, time),
                            ('detector', 'monitor', 'time')):
            if isinstance(arg, numbers.Number):
                arg = np.array([arg] * n_dim)
            setattr(self, key, np.array(arg))

        self.m0 = np.nanmax(self.monitor)
        self.t0 = np.nanmax(self.time)

        self.time_norm = time_norm

        if error is not None:
            self.error = error

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
            monitor = self.monitor
            detector = self.detector

            output = {'Q': right.Q, 'detector': np.negative(right.detector),
                      'monitor': right.monitor, 'time': right.time}
            return self.combine_data(output, ret=True)
        except AttributeError:
            raise AttributeError('Data types cannot be combined')

    def __mul__(self, right):
        self.detector = self.detector * right
        return self

    def __div__(self, right):
        self.detector = self.detector / right
        return self

    def __truediv__(self, right):
        self.detector = self.detector / right
        return self

    def __floordiv__(self, right):
        self.detector = self.detector // right
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
        r'''Returns error of monitor or time normalized intensity

        '''
        try:
            if self._err is not None:
                err = self._err
            else:
                err = np.sqrt(self.detector)
        except AttributeError:
            self._err = None
            err = np.sqrt(self.detector)

        if self.time_norm:
            if self.t0 == 0:
                self.t0 = np.nanmax(self.time)
            return err / self.time * self.t0
        else:
            if self.m0 == 0:
                self.m0 = np.nanmax(self.monitor)
            return err / self.monitor * self.m0

    @error.setter
    def error(self, value):
        r'''Set error in detector counts
        '''
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.detector.shape[0])

        if value.shape != self.detector.shape:
            raise ValueError('''Input value must have the shape ({0},) or be \
                                a float.'''.format(self.detector.shape[0]))
        self._err = value

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

        '''
        Q = self.Q.copy()
        detector = self.detector.copy()
        monitor = self.monitor.copy()
        time = self.time.copy()

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
                    arg[key] = np.delete(arg[key],
                                         (np.array(combine,
                                                   dtype=np.int64)[:, 0],), 0)

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

    def subtract_background(self, background_data, ret=True):
        r'''Subtract background data.

        Parameters
        ----------
        background_data : Data object
            Data object containing the data wishing to be subtracted

        ret : bool, optional
            Set False if background should be subtracted in place.
            Default: True

        Returns
        -------
        data : Data object
            Data object contained subtracted data

        '''
        pass

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
        error = np.empty(Q_chunk.shape[0])

        for i, _Q_chunk in enumerate(Q_chunk):
            _Q = self.Q
            _mon = self.monitor
            _det = self.detector
            _tim = self.time
            _err = self.error

            for j in range(_Q.shape[1]):
                _order = np.lexsort([_Q[:, j - n] for n
                                     in reversed(range(_Q.shape[1]))])
                _Q = _Q[_order]
                _mon = _mon[_order]
                _det = _det[_order]
                _tim = _tim[_order]
                _err = _err[_order]

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
                    _err = _err[chunk0:chunk1]

            monitor[i] = np.average(_mon[chunk0:chunk1])
            detector[i] = np.average(_det[chunk0:chunk1])
            time[i] = np.average(_tim[chunk0:chunk1])
            error[i] = np.sqrt(1 / np.sum(1 / _err[chunk0:chunk1] ** 2))
            # error[i] = 1 / np.sqrt(2) * np.average(_err[chunk0:chunk1])

        return (monitor, detector, time, error)

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
                _q, _qstep = (np.array([np.average(arg[:2])]),
                              (arg[1] - arg[0]))
            else:
                _q, _qstep = np.linspace(arg[0], arg[1], arg[2], retstep=True)
            q += _q,
            qstep += _qstep,

        self._qstep = qstep

        Q = np.meshgrid(*q)
        Q = np.vstack((item.flatten() for item in Q)).T

        nprocs = cpu_count()
        Q_chunks = [Q[n * Q.shape[0] // nprocs:(n + 1) * Q.shape[0] // nprocs]
                    for n in range(nprocs)]
        pool = Pool(processes=nprocs)
        outputs = pool.map(_call_bin_parallel, zip([self] * len(Q_chunks), Q_chunks))
        pool.close()

        monitor, detector, time, error = (np.concatenate(arg) for arg in zip(*outputs))

        return Data(Q=Q, monitor=monitor, detector=detector, time=time,
                    m0=self.m0, t0=self.t0, error=error)
