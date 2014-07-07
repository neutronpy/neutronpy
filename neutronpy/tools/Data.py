import numpy as np
import matplotlib.pyplot as plt  # @UnusedImport
from scipy.interpolate import griddata  # @UnusedImport


class Data(object):
    '''Data class for handling multi-dimensional TAS data
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_file(self, *files, mode='', **kwargs):
        '''Loads one or more files in either HFIR or NCNR formats
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

                *args, = np.loadtxt(filename, unpack=True, dtype=np.float64)  # @IgnorePep8

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
                *args, = np.loadtxt(filename, unpack=True, dtype=np.float64, skiprows=12)  # @IgnorePep8

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
        '''Internal class for constructing Q[q, hw] from h, k, l, and energy
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
        '''Returns monitor normalized intensity
        '''
        try:
            m0 = kwargs['m0']
        except:
            try:
                m0 = self.m0
            except:
                m0 = np.nanmax(self.monitor)
                self.m0 = m0

        return self.detector / self.monitor * m0

    def error(self, **kwargs):
        '''Returns square-root error of monitor normalized intensity
        '''
        return np.sqrt(self.intensity(**kwargs))

    def combine_data(self, *args, **kwargs):
        '''Combines multiple data sets
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

        monitor, detector, temp = [np.zeros(Q.shape[0])] * 3

        for i in range(Q.shape[0]):
            to_bin = np.where(np.abs((self.Q[:, 0] - Q[i, 0]) ** 2 / (qstep[0] / 2.) ** 2 +
                                     (self.Q[:, 1] - Q[i, 1]) ** 2 / (qstep[1] / 2.) ** 2 +
                                     (self.Q[:, 2] - Q[i, 2]) ** 2 / (qstep[2] / 2.) ** 2 +
                                     (self.Q[:, 3] - Q[i, 3]) ** 2 / (qstep[3] / 2.) ** 2) < 1.)
            if to_bin[0]:
                monitor[i] = np.average(self.monitor[to_bin])
                detector[i] = np.average(self.detector[to_bin])
                temp[i] = np.average(self.temp[to_bin])

        return Q, monitor, detector, temp

    def integrate(self, bounds=None, **kwargs):
        '''Returns the integrated intensity within given bounds
        '''
        if hasattr(kwargs, 'bounds'):
            tofit = np.where(kwargs['bounds'])
            print(tofit)
        else:
            np.sum()

    def mean_squared_position(self, bounds=None, **kwargs):
        '''Returns the mean-squared position of a peak within the given bounds
        '''
        pass

    def width(self, bounds=None, **kwargs):
        '''Returns the width of a peak within the given bounds
        '''
        pass
