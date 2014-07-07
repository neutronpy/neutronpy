from ..constants import boltzmann_meV_K
import numpy as np
import matplotlib.pyplot as plt  # @UnusedImport
from scipy.interpolate import griddata  # @UnusedImport


class Data(object):
    '''Data class for handling multi-dimensional TAS data. If input file type is not supported, data can be entered manually.

    Parameters
    ----------
    h : ndarray, optional
        Array of Q$_x$ in reciprocal lattice units.

    k : ndarray, optional
        Array of Q$_y$ in reciprocal lattice units.

    l : ndarray, optional
        Array of Q$_z$ in reciprocal lattice units.

    e : ndarray, optional
        Array of $\hbar \omega$ in meV.

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

    def load_file(self, *files, mode, **kwargs):
        '''Loads one or more files in either HFIR or NCNR formats

        Parameters
        ----------
        files : string
            A file or non-keyworded list of files containing data for input.

        mode : string
            Specify file type (HFIR | NCNR). Currently only file types supported.

        Returns
        -------
        None

        '''
        if mode == 'HFIR':
            keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'monitor': 'monitor', 'detector': 'detector', 'temp': 'temp'}
            for filename in files:
                output = {}
                with open(filename) as f:
                    for line in f:
                        if 'col_headers' in line:
                            *args, = next(f).split()  # @IgnorePep8
                            headers = [head.replace('.', '') for head in args[1:]]

                * args, = np.loadtxt(filename, unpack=True, dtype=np.float64)  # @IgnorePep8

                for key, value in keys.items():
                    output[key] = args[headers.index(value)]

                if not hasattr(self, 'Q'):
                    for key, value in output.items():
                        setattr(self, key, value)
                    self.Q = self.build_Q(**kwargs)
                else:
                    output['Q'] = self.build_Q(output=output, **kwargs)
                    self.combine_data(output)

        if mode == 'NCNR':
            keys = {'h': 'Qx', 'k': 'Qy', 'l': 'Qz', 'e': 'E', 'monitor': 'monitor', 'detector': 'Counts', 'temp': 'Tact'}
            for filename in files:
                output = {}
                with open(filename) as f:
                    for i, line in enumerate(f):
                        if i == 0:
                            self.length = int(line.split()[-2])
                            self.m0 = float(line.split()[-5])
                        if 'Q(x)' in line:
                            *args, = line.split()  # @IgnorePep8
                            headers = [head.replace('(', '').replace(')', '').replace('-', '') for head in args]
                * args, = np.loadtxt(filename, unpack=True, dtype=np.float64, skiprows=12)  # @IgnorePep8

                for key, value in keys.items():
                    output[key] = args[headers.index(value)]

                if not self.Q:
                    for key, value in output.items():
                        setattr(self, key, value)
                    self.Q = self.build_Q(**kwargs)
                else:
                    output['Q'] = self.build_Q(output=output, **kwargs)
                    self.combine_data(output)

    def build_Q(self, **kwargs):
        '''Internal method for constructing Q[q, hw] from h, k, l, and energy

        Parameters
        ----------
        output : dictionary, optional
            A dictionary of the h, k, l, and e arrays to form into a column oriented array

        Returns
        -------
        Q : ndarray, shape (N, 4,)
            Returns Q (h, k, l, e) in a column oriented array.

        '''
        args = ()
        if hasattr(kwargs, 'output'):
            for i in ['h', 'k', 'l', 'e']:
                args += (getattr(kwargs['output'], i),)
        else:
            for i in ['h', 'k', 'l', 'e']:
                args += (getattr(self, i),)

        return np.vstack((item.flatten() for item in args)).T

    def intensity(self, **kwargs):
        '''Returns the monitor normalized intensity

        Parameters
        ----------
        m0 : float, optional
            Desired monitor to normalize the intensity

        Returns
        -------
        intensity : ndarray
            The monitor normalized intensity scaled by m0

        '''
        try:
            m0 = kwargs['m0']
        except:
            try:
                m0 = self.m0
            except:
                self.m0 = m0 = np.nanmax(self.monitor)

        return self.detector / self.monitor * m0

    def error(self, **kwargs):
        '''Returns square-root error of monitor normalized intensity

        Parameters
        ----------
        m0 : float, optional
            Desired monitor to normalize the intensity

        Returns
        -------
        error : ndarray
            The square-root error of the monitor normalized intensity

        '''
        return np.sqrt(self.intensity(**kwargs))

    def detailed_balance_factor(self, **kwargs):
        '''Returns the detailed balance factor (sometimes called the Bose factor)

        Parameters
        ----------
        temp : float, optional
            If not already a property of the class, the sample temperature can be specified as a float.

        Returns
        -------
        dbf : ndarray
            The detailed balance factor (temperature correction)

        '''
        try:
            self.temps = np.ones(self.Q.shape[0]) * kwargs['temp']
        except:
            pass

        return np.exp(-self.Q[3] / boltzmann_meV_K / self.temps)

    def combine_data(self, *args, **kwargs):
        '''Combines multiple data sets

        Parameters
        ----------
        args : dictionary of ndarrays
            A dictionary (or multiple) of the data that will be added to the current data, with keys:
                * Q : ndarray : [h, k, l, e] with shape (N, 4,)
                * monitor : ndarray : shape (N,)
                * detector : ndarray : shape (N,)
                * temps : ndarray : shape (N,)

        Returns
        -------
        None
            Adds the data to the class, provides no returns.

        '''
        for arg in args:
            combine = []
            for i in range(arg['Q'].shape[0]):
                for j in range(self.Q.shape[0]):
                    if np.all(self.Q[j, :] == arg['Q'][i, :]):
                        combine.append((i, j))

            for item in combine:
                self.monitor[item[0]] += arg['monitor'][item[1]]
                self.detector[item[0]] += arg['detector'][item[1]]

            for item in combine:
                for key in ['Q', 'monitor', 'detector', 'temp']:
                    np.delete(arg[key], item[0], 0)

            np.concatenate((self.Q, arg['Q']))
            np.concatenate((self.detector, arg['detector']))
            np.concatenate((self.monitor, arg['monitor']))
            np.concatenate((self.temp, arg['temp']))

            order = np.lexsort((self.Q[:, 0], self.Q[:, 1], self.Q[:, 2], self.Q[:, 3]))
            self.Q = self.Q[order]
            self.monitor = self.monitor[order]
            self.detector = self.detector[order]
            self.temp = self.temp[order]

            self.h = self.Q[:, 0]
            self.k = self.Q[:, 1]
            self.l = self.Q[:, 2]
            self.e = self.Q[:, 3]

    def bin(self, *args, **kwargs):
        '''Rebin the data into the specified shape.
        '''

        q, qstep = (), ()
        for arg in args:
            if arg[2] == 1:
                _q, _qstep = (np.array([(arg[1] - arg[0]) / 2.]), 1)
            else:
                _q, _qstep = np.linspace(arg[0], arg[1], arg[2], retstep=True)
            q += _q
            qstep += _qstep

        Q = np.meshgrid(*q)
        Q = np.vstack((item.flatten() for item in Q)).T

        monitor, detector, temps = [np.zeros(Q.shape[0])] * 3

        for i in range(Q.shape[0]):
            to_bin = np.where(np.abs((self.Q[:, 0] - Q[i, 0]) ** 2 / (qstep[0] / 2.) ** 2 +
                                     (self.Q[:, 1] - Q[i, 1]) ** 2 / (qstep[1] / 2.) ** 2 +
                                     (self.Q[:, 2] - Q[i, 2]) ** 2 / (qstep[2] / 2.) ** 2 +
                                     (self.Q[:, 3] - Q[i, 3]) ** 2 / (qstep[3] / 2.) ** 2 +
                                     (self.temps - Q[i, 4]) ** 2 / (qstep[4] / 2.)) < 1.)
            if to_bin[0]:
                monitor[i] = np.average(self.monitor[to_bin])
                detector[i] = np.average(self.detector[to_bin])
                temps[i] = np.average(self.temp[to_bin])

        return Q, monitor, detector, temps

    def integrate(self, **kwargs):
        '''Returns the integrated intensity within given bounds
        '''
        if hasattr(kwargs, 'bounds'):
            tofit = np.where(kwargs['bounds'])
            print(tofit)
        else:
            np.sum()

    def mean_squared_position(self, **kwargs):
        '''Returns the mean-squared position of a peak within the given bounds
        '''
        if hasattr(kwargs, 'bounds'):
            tofit = np.where(kwargs['bounds'])
            print(tofit)
        else:
            np.sum()

    def width(self, **kwargs):
        '''Returns the width of a peak within the given bounds
        '''
        if hasattr(kwargs, 'bounds'):
            tofit = np.where(kwargs['bounds'])
            print(tofit)
        else:
            np.sum()

    def plot(self, **kwargs):
        axes = ['x', 'y', 'z', 'w']
        options = ['h', 'k', 'l', 'e', 'temp', 'intensity']

        to_bin = ()
        for opt in options[:4]:
            for axis in axes:
                if getattr(kwargs, axis)[0] == opt:
                    to_bin += (getattr(kwargs, axis)[1:],)

        Q, monitor, detector, temps = self.bin(*to_bin)
