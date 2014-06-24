import numpy as np
import matplotlib.pyplot as plt  # @UnusedImport


class Data(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_file(self, filename, mode='auto', **kwargs):
        if mode == 'auto':
            *args, = np.loadtxt('scan.dat', unpack=True, dtype=np.float64)  # @IgnorePep8
            self.length = args[0].shape[0]

        if mode == 'HFIR':
            with open(filename) as f:
                for line in f:
                    if 'col_headers' in line:
                        *args, = next(f).split()  # @IgnorePep8
                        headers = [head.replace('.', '') for head in args[1:]]

            * args, = np.loadtxt(filename, unpack=True, dtype=np.float64)  # @IgnorePep8
            self.length = args[0].shape[0]
            for key, value in zip(headers, args):
                setattr(self, key, value)

        if mode == 'iexy':
            (self.inten, self.err, self.x, self.y) = np.loadtxt(filename, unpack=True, dtype=np.float64)

        try:
            self.keys = kwargs.get('keys', None)
            self.build_data()
        except AttributeError:
            raise AttributeError('Keys are not defined.')

    def build_data(self):
        '''{'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'monitor': 'monitor', 'detector': 'detector'}
        '''
        args = ()
        for i in ['h', 'k', 'l', 'e']:
            if isinstance(self.keys[i], (int, float)):
                args += (np.ones(self.length) * self.keys[i],)
            elif type(self.keys[i]) is str:
                args += (getattr(self, self.keys[i]),)
            else:
                raise ValueError('Provided key is not an understood value.')

        self.Q = np.vstack((item.flatten() for item in args)).T

        invars = ['monitor', 'detector', 'flux', 'intensity', 'error', 'temperature']
        outvars = ['mon', 'det', 'flux', 'inten', 'err', 'temp']
        for varin, varout in zip(invars, outvars):
            try:
                setattr(self, varout, getattr(self, self.keys[varin]))
            except (KeyError, AttributeError):
                pass

        if not hasattr(self, 'err'):
            for i in ['inten', 'detector']:
                try:
                    self.err = np.sqrt(getattr(self, i))
                except (KeyError, AttributeError):
                    pass

    def normalize_to_monitor(self, monitor):
        try:
            self.inten = self.detector / self.monitor * monitor
            self.err = self.err / self.monitor * monitor
        except AttributeError:
            raise

    def combine_data(self, inp, mode='TAS'):
        if type == 'TAS':
            pass
        if type == 'TOF':
            pass

    def bin(self, *args):
        '''Rebin the data into the specified shape.
        '''
        if not hasattr(self, 'Q'):
            try:
                self.build_Q(self.h, self.k, self.l, self.e)
            except AttributeError:
                raise AttributeError('Q has not been built, use build_Q(h, k, l, e) class method.')

        q, qstep = (), ()
        for arg in args:
            if arg[2] == 1:
                _q, _qstep = (np.array([(arg[1] - arg[0]) / 2.]), 1)
            else:
                _q, _qstep = np.linspace(arg[0], arg[1], arg[2], retstep=True)
            q += _q
            qstep += _qstep

        qxstep, qystep, qzstep, wstep = qstep

        q = np.meshgrid(*q)
        Q = np.vstack((item.flatten() for item in q)).T

        mon = np.zeros(Q.shape[0])
        count = np.zeros(Q.shape[0])

        for i in range(Q.shape[0]):
            to_bin = np.where(np.abs((self.Q[:, 0] - Q[i, 0]) ** 2 / (qxstep / 2.) ** 2 +
                                     (self.Q[:, 1] - Q[i, 1]) ** 2 / (qystep / 2.) ** 2 +
                                     (self.Q[:, 2] - Q[i, 2]) ** 2 / (qzstep / 2.) ** 2 +
                                     (self.Q[:, 3] - Q[i, 3]) ** 2 / (wstep / 2.) ** 2) < 1.)
            if to_bin[0]:
                mon[i] = np.average(self.mon[to_bin])
                count[i] = np.average(self.count[to_bin])
