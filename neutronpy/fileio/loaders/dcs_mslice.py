import numpy as np
from ...data import Data
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class DcsMslice(Data):
    r'''Loads DCS_MSLICE (NCNR) exported ascii data files.

    '''
    def __init__(self):
        super(DcsMslice, self).__init__()

    def load(self, filename, **kwargs):
        r'''Loads the DCS_MSLICE (NCNR) exported ascii data files, including
        SPE, IEXY, XYE, and XYIE. NOTE: Will NOT load DAVE format files.
        DAVE files are propietary IDL formatted save files.

        Parameters
        ----------
        filename : str
            Path to file to open

        '''
        load_file = {'iexy': self.load_iexy,
                     'xyie': self.load_xyie,
                     'spe': self.load_spe,
                     'xye': self.load_xye}

        load_file[self.determine_subtype(filename)](filename)

    def determine_subtype(self, filename):
        try:
            with open(filename) as f:
                first_line = f.readline()
                second_line = f.readline()
                if filename[-4:] == 'iexy' or len(first_line.split()) == 4:
                    return 'iexy'
                elif filename[-4:] == 'xyie' or (len(first_line.split()) == 2 and len(second_line) == 0):
                    return 'xyie'
                elif filename[-3:] == 'spe' or (len(first_line.split()) == 2 and '###' in second_line):
                    return 'spe'
                elif filename[-3:] == 'xye' or len(first_line.split()) == 3:
                    return 'xye'
                else:
                    raise IOError('Cannot determine filetype!')
        except IOError:
            raise

    def load_iexy(self, filename):
        i, e, x, y = np.loadtxt(filename, unpack=True)
        self._data = OrderedDict(intensity=i, error=e, x=x, y=y, monitor=np.ones(len(i)), time=np.ones(len(i)))
        self.data_keys = {'intensity': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = e

    def load_spe(self, filename):
        pass

    def load_xyie(self, filename):
        with open(filename) as f:
            data = []
            for line in f:
                data.append(line.replace('\n', '').split())

        shape = tuple(int(i) for i in data[0])
        x = np.squeeze(np.array(data[2:2 + shape[0]]).astype(float))
        y = np.squeeze(np.array(data[3 + shape[0]:3 + shape[0] + shape[1]]).astype(float))
        i = np.array(data[4 + np.sum(shape):4 + np.sum(shape) + shape[1]]).astype(float)
        e = np.array(data[5 + np.sum(shape) + shape[1]:5 + np.sum(shape) + 2 * shape[1]]).astype(float)

        if x.shape[0] == shape[0] and y.shape[0] == shape[1] and i.T.shape == shape and e.T.shape == shape:
            X, Y = np.meshgrid(x, y)
            self._data = OrderedDict(intensity=i.flatten(),
                                     error=e.flatten(),
                                     x=X.flatten(),
                                     y=X.flatten(),
                                     monitor=np.ones(len(i)).flatten(),
                                     time=np.ones(len(i)).flatten())
            self.data_keys = {'intensity': 'intensity', 'monitor': 'monitor', 'time': 'time'}
            self._err = e
        else:
            raise ValueError('File was not loaded correctly!')

    def load_xye(self, filename):
        x, y, e = np.loadtxt(filename, unpack=True)
        self._data = OrderedDict(intensity=y, error=e, x=x, monitor=np.ones(len(y)), time=np.ones(len(y)))
        self.data_keys = {'intensity': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = e
