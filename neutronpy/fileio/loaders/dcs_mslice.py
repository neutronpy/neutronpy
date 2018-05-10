from collections import OrderedDict

import numpy as np

from ...data import Data


class DcsMslice(Data):
    r"""Loads DCS_MSLICE (NCNR) exported ascii data files.

    """

    def __init__(self):
        super(DcsMslice, self).__init__()

    def load(self, filename, **kwargs):
        r"""Loads the DCS_MSLICE (NCNR) exported ascii data files, including
        SPE, IEXY, XYE, and XYIE. NOTE: Will NOT load DAVE format files.
        DAVE files are propietary IDL formatted save files.

        Parameters
        ----------
        filename : str
            Path to file to open

        """
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
        self.data_keys = {'detector': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = e

    def load_spe(self, filename):
        with open(filename) as f:
            data = []
            for line in f:
                if '###' not in line:
                    _line = line.replace('\n', '').replace('-', ' -').replace('e -', 'e-').split()
                else:
                    _line = line.replace('\n', '')
                data.append(_line)

            shape = tuple(int(i) for i in data[0])
            i = 1
            col_headers = []
            for line in data:
                if '###' in line:
                    col_headers.append(line.split('### ')[-1])
                    if len(col_headers) == 4:
                        break

            x = []
            for n, line in enumerate(data[2:]):
                if '###' in line or len(x) >= shape[0]:
                    line_num = 3 + n
                    break
                x.extend(line)

            y = []
            for n, line in enumerate(data[line_num:]):
                if '###' in line or len(x) >= shape[1]:
                    line_num += n + 1
                    break
                y.extend(line)

            x = np.squeeze(np.array(x).astype(float))[:shape[0]]
            y = np.squeeze(np.array(y).astype(float))[:shape[1]]
            X, Y = np.meshgrid(x, y)

            _temp_int_err = []
            for i in range(shape[0] * 2):
                _temp_data = []
                for n, line in enumerate(data[line_num:]):
                    if '###' in line:
                        line_num += n + 1
                        break
                    else:
                        _temp_data.extend(line)
                _temp_int_err.append(np.squeeze(np.array(_temp_data).astype(float))[:shape[1]])
            intensity = np.array(_temp_int_err[0::2]).T
            err = np.array(_temp_int_err[1::2]).T

            self._data = OrderedDict(intensity=intensity.flatten(),
                                     error=err.flatten(),
                                     x=X.flatten(),
                                     y=X.flatten(),
                                     monitor=np.ones(len(intensity)).flatten(),
                                     time=np.ones(len(intensity)).flatten())
            self.data_keys = {'detector': 'intensity', 'monitor': 'monitor', 'time': 'time'}
            self._err = err

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
            self.data_keys = {'detector': 'intensity', 'monitor': 'monitor', 'time': 'time'}
            self._err = e
        else:
            raise ValueError('File was not loaded correctly!')

    def load_xye(self, filename):
        x, y, e = np.loadtxt(filename, unpack=True)
        self._data = OrderedDict(intensity=y, error=e, x=x, monitor=np.ones(len(y)), time=np.ones(len(y)))
        self.data_keys = {'detector': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = e
