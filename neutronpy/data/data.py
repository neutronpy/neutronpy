# -*- coding: utf-8 -*-
r"""Data handling

"""
import copy
import numbers
import warnings
import numpy as np
from collections import OrderedDict
from multiprocessing import cpu_count, Pool  # @UnresolvedImport
from .analysis import Analysis
from .plot import PlotData



def _call_bin_parallel(arg, **kwarg):
    r"""Wrapper function to work around pickling problem in Python 2.7
    """
    return Data._bin_parallel(*arg, **kwarg)


class Data(PlotData, Analysis):
    u"""Data class for handling multi-dimensional scattering data. If input
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
    plot_volume
    plot_contour
    plot_line
    get_keys
    get_bounds

    """

    def __init__(self, Q=None, h=0., k=0., l=0., e=0., temp=0., detector=0., monitor=0., error=None, time=0.,
                 time_norm=False, **kwargs):
        self._data = OrderedDict()
        self.data_keys = {'monitor': 'monitor', 'detector': 'detector', 'time': 'time'}
        self.Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'temp'}

        if Q is None:
            try:
                n_dim = max([len(item) for item in (h, k, l, e, temp, detector, monitor, time) if
                             not isinstance(item, numbers.Number)])
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
        except TypeError:
            raise

    def __iadd__(self, right):
        try:
            self.combine_data(right, ret=False)
        except TypeError:
            raise

    def __sub__(self, right):
        try:
            return self.subtract_background(right, ret=True)
        except (TypeError, ValueError):
            raise

    def __isub__(self, right):
        try:
            self.subtract_background(right, ret=False)
        except (TypeError, ValueError):
            raise

    def __mul__(self, right):
        temp_obj = copy.deepcopy(self)
        temp_obj.detector = self.detector * right
        return temp_obj

    def __imul__(self, right):
        self.detector *= right

    def __div__(self, right):
        temp_obj = copy.deepcopy(self)
        temp_obj.detector = self.detector / right
        return temp_obj

    def __idiv__(self, right):
        self.detector /= right

    def __truediv__(self, right):
        temp_obj = copy.deepcopy(self)
        temp_obj.detector = self.detector / right
        return temp_obj

    def __itruediv__(self, right):
        self.detector /= right

    def __floordiv__(self, right):
        temp_obj = copy.deepcopy(self)
        temp_obj.detector = self.detector // right
        return temp_obj

    def __ifloordiv__(self, other):
        self.detector //= right

    def __pow__(self, right):
        temp_obj = copy.deepcopy(self)
        temp_obj.detector **= right
        return temp_obj

    def __eq__(self, right):
        if not np.all(sorted(list(self.data.keys())) == sorted(list(right.data.keys()))):
            return False

        for key, value in self.data.items():
            if not np.all(value == right.data[key]):
                return False

        return True

    def __ne__(self, right):
        return not self.__eq__(right)

    @property
    def Q(self):
        r"""Returns a Q matrix with columns h,k,l,e,temp
        """
        return np.vstack((self.data[self.Q_keys[i]].flatten() for i in ['h', 'k', 'l', 'e', 'temp'])).T

    @Q.setter
    def Q(self, value):
        for col, key in zip(value.T, ['h', 'k', 'l', 'e', 'temp']):
            self._data[self.Q_keys[key]] = col

    @property
    def detector(self):
        r"""Returns the raw counts on the detector
        """
        return self.data[self.data_keys['detector']]

    @detector.setter
    def detector(self, value):
        self.data[self.data_keys['detector']] = value

    @property
    def monitor(self):
        r"""Returns the monitor
        """
        return self.data[self.data_keys['monitor']]

    @monitor.setter
    def monitor(self, value):
        self.data[self.data_keys['monitor']] = value

    @property
    def time(self):
        r"""Returns the time measured
        """
        return self.data[self.data_keys['time']]

    @time.setter
    def time(self, value):
        self.data[self.data_keys['time']] = value

    @property
    def h(self):
        r"""Returns lattice parameter q\ :sub:`x`\ , *i.e.* h

        Equivalent to Q[:, 0]
        """
        return self.data[self.Q_keys['h']]

    @h.setter
    def h(self, value):
        r"""Set h to appropriate column of Q
        """
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.data[self.data_keys['detector']].shape[0])

        if value.shape != self.data[self.data_keys['detector']].shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(
                self.data[self.data_keys['detector']].shape))

        else:
            self.data[self.Q_keys['h']] = np.array(value)

    @property
    def k(self):
        r"""Returns lattice parameter q\ :sub:`y`\ , *i.e.* k

        Equivalent to Q[:, 1]
        """
        return self.data[self.Q_keys['k']]

    @k.setter
    def k(self, value):
        r"""Set k to appropriate column of Q
        """
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.data[self.data_keys['detector']].shape[0])

        if value.shape != self.data[self.data_keys['detector']].shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(
                self.data[self.data_keys['detector']].shape))

        else:
            self.data[self.Q_keys['k']] = np.array(value)

    @property
    def l(self):
        r"""Returns lattice parameter q\ :sub:`z`\ , *i.e.* l

        Equivalent to Q[:, 2]
        """
        return self.data[self.Q_keys['l']]

    @l.setter
    def l(self, value):
        r"""Set l to appropriate column of Q
        """
        if isinstance(value, numbers.Number):
            value = value = np.array([value] * self.data[self.data_keys['detector']].shape[0])

        if value.shape != self.data[self.data_keys['detector']].shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(
                self.data[self.data_keys['detector']].shape))

        else:
            self.data[self.Q_keys['l']] = np.array(value)

    @property
    def e(self):
        r"""Returns energy transfer

        Equivalent to Q[:, 3]
        """
        return self.data[self.Q_keys['e']]

    @e.setter
    def e(self, value):
        r"""Set e to appropriate column of Q
        """
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.data[self.data_keys['detector']].shape[0])

        if value.shape != self.data[self.data_keys['detector']].shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(
                self.data[self.data_keys['detector']].shape))

        else:
            self.data[self.Q_keys['e']] = np.array(value)

    @property
    def temp(self):
        r"""Returns temperature

        Equivalent to Q[:, 4]
        """
        return self.data[self.Q_keys['temp']]

    @temp.setter
    def temp(self, value):
        r"""Set temp to appropriate column of Q
        """
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.data[self.data_keys['detector']].shape[0])

        if value.shape != self.data[self.data_keys['detector']].shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(
                self.data[self.data_keys['detector']].shape))

        else:
            self.data[self.Q_keys['temp']] = np.array(value)

    @property
    def intensity(self):
        r"""Returns the monitor or time normalized intensity

        """

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
        r"""Returns error of monitor or time normalized intensity

        """
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
        r"""Set error in detector counts
        """
        if isinstance(value, numbers.Number):
            value = np.array([value] * self.detector.shape[0])

        if value.shape != self.detector.shape:
            raise ValueError("""Input value must have the shape ({0},) or be a float.""".format(self.detector.shape[0]))

        self._err = value

    @property
    def data(self):
        r"""Returns all of the raw data in column format
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def data_columns(self):
        r"""Returns a list of the raw data columns
        """
        return list(self.data.keys())

    def combine_data(self, obj, **kwargs):
        r"""Combines multiple data sets

        Parameters
        ----------
        obj : Data_object
            Data_object with equivalent data columns

        tols : ndarray or float, optional
            Tolerances for combining two data sets. Default: 5e-4.

        ret : bool, optional
            Return the combined data set, or merge. Default: False

        """
        if not isinstance(obj, Data):
            raise TypeError('You can only combine two Data objects: input object is the wrong format!')

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
                orig_vals = np.array(
                    [val[j] for k, val in self._data.items() if k not in list(self.data_keys.values())])
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

    def subtract_background(self, background_data, x=None, ret=True):
        r"""Subtract background data.

        Parameters
        ----------
        background_data : Data object
            Data object containing the data wishing to be subtracted

        x : str, optional
            data_column key of x-axis values over which background should be
            subtracted. Used for cases where background data is not taken at
            exactly same points as data being subtracted. Default: None

        ret : bool, optional
            Set False if background should be subtracted in place.
            Default: True

        Returns
        -------
        data : Data object
            Data object contained subtracted data

        """
        if not isinstance(background_data, Data):
            raise TypeError('You can only combine two Data objects: input object is the wrong format!')

        if self.time_norm != background_data.time_norm:
            warnings.warn(
                'Normalization of detector is different. One is normalized to time, and the other to monitor.')

        if x is None:
            try:
                _new_intensity = self.intensity - background_data.intensity
                _new_error = np.sqrt(
                    np.array([np.max([err1 ** 2, err2 ** 2]) for err1, err2 in zip(self.error, background_data.error)]))
            except ValueError:
                raise ValueError(
                    'Data objects are incompatible shapes: try subtract_background method for more options')
        else:
            try:
                bg_x = background_data.data[x]
                self_x = self.data[x]
            except KeyError:
                try:
                    bg_x = background_data.data[self.Q_keys[x]]
                    self_x = self.data[self.Q_keys[x]]
                except (AttributeError, KeyError):
                    raise KeyError('Invalid key for data_column.')

            try:
                from scipy.interpolate import griddata

                bg_intensity_grid = griddata(bg_x, background_data.intensity, self_x, method='nearest')
                bg_error_grid = np.sqrt(griddata(bg_x, background_data.error ** 2, self_x, method='nearest'))
            except ImportError:
                warnings.warn('Background subtraction failed. Scipy Import Error, use more recent version of Python')

                if ret:
                    return self

            _new_intensity = self.intensity - bg_intensity_grid.flatten()
            _new_error = np.sqrt(
                np.array([np.max([err1 ** 2, err2 ** 2]) for err1, err2 in zip(self.error, bg_error_grid.flatten())]))

        _sub_data = copy.copy(self.data)
        _sub_data[self.data_keys['detector']] = _new_intensity
        _sub_data[self.data_keys['monitor']] = np.ones(_new_intensity.shape)
        _sub_data[self.data_keys['time']] = np.ones(_new_intensity.shape)

        if ret:
            data_obj = copy.deepcopy(self)
            data_obj.t0 = 1
            data_obj.m0 = 1
            data_obj._err = _new_error
            data_obj._data = _sub_data
            return data_obj
        else:
            self.t0 = 1
            self.m0 = 1
            self._err = _new_error
            self._data = _sub_data

    def _bin_parallel(self, Q_chunk):
        r"""Performs binning by finding data chunks to bin together.
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

        """
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
        r"""Rebin the data into the specified shape.

        Parameters
        ----------
        to_bin : dict
            A dictionary containing information about which data_column
            should be binned in the following format:

                `'key': [lower_bound, upper_bound, num_points]`

            Any key in `data_column` is a valid key. Those keys from
            `data_column` not included in `to_bin` are averaged.

        build_hkl : bool, optional
            Toggle to build hkle. Must already have hkle built in object you
            are binning. Default: True

        Returns
        -------
        binned_data : :class:`.Data` object
            The resulting data object with values binned to the specified
            bounds

        """
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
