# -*- coding: utf-8 -*-
r'''Data handling

'''
import copy
from multiprocessing import cpu_count, Pool  # @UnresolvedImport
import numbers
import numpy as np
from .analysis import Analysis
from .plot import PlotData
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


def _call_bin_parallel(arg, **kwarg):
    r'''Wrapper function to work around pickling problem in Python 2.7
    '''
    return Data._bin_parallel(*arg, **kwarg)


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
    def __init__(self, Q=None, h=0., k=0., l=0., e=0., temp=0., detector=0., monitor=0., error=None, time=0., time_norm=False, **kwargs):
        self._data = OrderedDict()
        self.data_keys = {'monitor': 'monitor', 'detector': 'detector', 'time': 'time'}
        self.Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'temp'}

        if Q is None:
            try:
                n_dim = max([len(item) for item in (h, k, l, e, temp, detector, monitor, time) if not isinstance(item, numbers.Number)])
            except (ValueError, UnboundLocalError):
                n_dim = 1

            self.Q = np.empty((n_dim, 5))

            for arg, key in zip((h, k, l, e, temp), ('h', 'k', 'l', 'e', 'temp')):
                if isinstance(arg, numbers.Number):
                    arg = np.array([arg] * n_dim)
                try:
                    self._data[self.Q_keys[key]] = np.array(arg)
                except (ValueError, KeyError):
                    raise
        else:
            self.Q = Q
            n_dim = Q.shape[1]

        for arg, key in zip((detector, monitor, time), ('detector', 'monitor', 'time')):
            if isinstance(arg, numbers.Number):
                arg = np.array([arg] * n_dim)
            self._data[self.data_keys[key]] = np.array(arg)

        self.m0 = np.nanmax(self.monitor)
        self.t0 = np.nanmax(self.time)

        self.time_norm = time_norm

        if error is not None:
            self.error = error

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __add__(self, right):
        try:
            return self.combine_data(right, ret=True)
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
    def Q(self):
        r'''Returns a Q matrix with columns h,k,l,e,temp
        '''
        return np.vstack((self.data[self.Q_keys[i]].flatten() for i in ['h', 'k', 'l', 'e', 'temp'])).T

    @Q.setter
    def Q(self, value):
        for col, key in zip(value.T, ['h', 'k', 'l', 'e', 'temp']):
            self._data[self.Q_keys[key]] = col

    @property
    def detector(self):
        r'''Returns the raw counts on the detector
        '''
        return self.data[self.data_keys['detector']]

    @detector.setter
    def detector(self, value):
        self.data[self.data_keys['detector']] = value

    @property
    def monitor(self):
        r'''Returns the monitor
        '''
        return self.data[self.data_keys['monitor']]

    @monitor.setter
    def monitor(self, value):
        self.data[self.data_keys['monitor']] = value

    @property
    def time(self):
        r'''Returns the time measured
        '''
        return self.data[self.data_keys['time']]

    @time.setter
    def time(self, value):
        self.data[self.data_keys['time']] = value

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
            value = np.array([value] * self.Q.shape[0])

        if value.shape != self.Q.shape[0]:
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.Q.shape[0]))
        else:
            self.data[self.Q_keys['h']] = np.array(value)

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
            value = np.array([value] * self.Q.shape[0])

        if value.shape != self.Q.shape[0]:
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.Q.shape[0]))
        else:
            self.data[self.Q_keys['k']] = np.array(value)

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
            value = value = np.array([value] * self.Q.shape[0])

        if value.shape != self.Q.shape[0]:
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.Q.shape[0]))
        else:
            self.data[self.Q_keys['l']] = np.array(value)

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
            value = np.array([value] * self.Q.shape[0])

        if value.shape != self.Q.shape[0]:
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.Q.shape[0]))
        else:
            self.data[self.Q_keys['e']] = np.array(value)

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
            value = np.array([value] * self.Q.shape[0])

        if value.shape != self.Q.shape[0]:
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.Q.shape[0]))
        else:
            self.data[self.Q_keys['temp']] = np.array(value)

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
            raise ValueError('''Input value must have the shape ({0},) or be a float.'''.format(self.detector.shape[0]))
        self._err = value

    @property
    def data(self):
        r'''Returns all of the raw data in column format
        '''
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def data_columns(self):
        r'''Returns a list of the raw data columns
        '''
        return list(self.data.keys())

    def combine_data(self, obj, **kwargs):
        r'''Combines multiple data sets

        Parameters
        ----------
        obj : Data_object
            Data_object with equivalent data columns

        tols : ndarray or float, optional
            Tolerances for combining two data sets. Default: 5e-4.

        ret : bool, optional
            Return the combined data set, or merge. Default: False

        '''
        if not isinstance(obj, Data):
            raise ValueError('You can only combine two Data objects: input object is the wrong format!')

        tols = np.array([5.e-4 for i in range(len(obj._data) - len(self.data_keys))])
        try:
            if kwargs['tols'] is not None:
                tols = np.array(kwargs['tols'])
        except KeyError:
            pass

        # combine
        _data_temp = copy.deepcopy(self._data)
        for i in range(len(obj._data[obj.data_keys['detector']])):
            new_vals = np.array([val[i] for k, val in obj._data.items() if k not in list(obj.data_keys.values())])
            for j in range(len(self._data[self.data_keys['detector']])):
                orig_vals = np.array([val[j] for k, val in self._data.items() if k not in list(self.data_keys.values())])
                if (np.abs(orig_vals - new_vals) <= tols).all():
                    for _key, _value in _data_temp.items():
                        if _key in list(self.data_keys.values()):
                            _data_temp[_key][j] += obj._data[_key][i]
                    break
            else:
                for _key, _value in _data_temp.items():
                    _data_temp[_key] = np.concatenate((_value, np.array([obj._data[_key][i]])))

        # sort
        ind = np.lexsort(tuple(value for key, value in _data_temp.items() if key not in list(self.data_keys.values())))
        _data = OrderedDict()
        for key, value in _data_temp.items():
            _data[key] = value[ind]

        if 'ret' in kwargs and kwargs['ret']:
            output = Data()
            output._data = _data
            return output
        else:
            self._data = _data

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
        error = np.empty(Q_chunk.shape[0])
        data_out = tuple(np.empty(Q_chunk.shape[0]) for key in self.data.keys() if key not in self.bin_keys)

        for i, _Q_chunk in enumerate(Q_chunk):
            _Q = np.vstack((self._data[key].flatten() for key in self.bin_keys)).T
            _data_out = tuple(value for key, value in self._data.items() if key not in self.bin_keys)
            _err = self.error

            above = _Q_chunk + np.array(self._qstep, dtype=float) / 2.
            below = _Q_chunk - np.array(self._qstep, dtype=float) / 2.

            bin_ind = np.where(((_Q <= above).all(axis=1) & (_Q >= below).all(axis=1)))

            if len(bin_ind[0]) > 0:
                for j in range(len(data_out)):
                    data_out[j][i] = np.average(_data_out[j][bin_ind])
                error[i] = np.sqrt(np.average(_err[bin_ind] ** 2))
            else:
                for j in range(len(data_out)):
                    data_out[j][i] = np.nan
                error[i] = np.nan

        return data_out + (error,)

    def bin(self, to_bin, build_hkl=True):
        r'''Rebin the data into the specified shape.

        Parameters
        ----------
        to_bin : dict
            A dictionary containing information about which data_column should
            be binned in the following format:

                `'key': [lower_bound, upper_bound, num_points]`

            Any data_column is a valid key. Any data_column key not included
            is ignored during the bin, and will not be returned in the new
            object.

        build_hkl : bool, optional
            Toggle to build hkle. Must already have hkle built in object you
            are binning. Default: True

        Returns
        -------
        binned_data : :class:`.Data` object
            The resulting data object with values binned to the specified bounds

        '''
        _bin_keys = list(to_bin.keys())
        if build_hkl:
            for key, value in self.Q_keys.items():
                if key in _bin_keys:
                    _bin_keys.remove(key)
                _bin_keys.append(value)

        self.bin_keys = copy.copy(_bin_keys)

        args = tuple()
        for key in self.bin_keys:
            try:
                args += to_bin[key],
            except KeyError:
                if key in self.Q_keys.values():
                    args += [self.data[key].min(), self.data[key].max(), 1],
                else:
                    raise KeyError

        q, qstep = tuple(), tuple()
        for arg in args:
            if arg[-1] == 1:
                _q, _qstep = (np.array([np.average(arg[:2])]), (arg[1] - arg[0]))
            else:
                _q, _qstep = np.linspace(arg[0], arg[1], arg[2], retstep=True)
            q += _q,
            qstep += _qstep,

        self._qstep = qstep

        Q = np.meshgrid(*q)
        Q = np.vstack((item.flatten() for item in Q)).T

        nprocs = cpu_count()
        Q_chunks = [Q[n * Q.shape[0] // nprocs:(n + 1) * Q.shape[0] // nprocs] for n in range(nprocs)]
        pool = Pool(processes=nprocs)
        outputs = pool.map(_call_bin_parallel, zip([self] * len(Q_chunks), Q_chunks))
        pool.close()

        _data_out = [np.concatenate(arg) for arg in zip(*outputs)]

        data_out = tuple()
        del_nan = np.where(np.isnan(_data_out[0]))
        for arg in _data_out:
            data_out += np.delete(arg, del_nan, axis=0),

        Q = np.delete(Q, del_nan, axis=0)

        _data = copy.copy(self._data)
        n = 0
        for key in _data.keys():
            if key not in self.bin_keys:
                _data[key] = data_out[n]
                n += 1
            else:
                _data[key] = Q[:, self.bin_keys.index(key)]

        output = copy.deepcopy(self)
        output._data = _data
        output._err = data_out[-1]

        return output
