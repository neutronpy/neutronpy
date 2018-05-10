import warnings
from collections import OrderedDict

import numpy as np

from ...data import Data
from ..instrument import load_instrument as load_instr


class Neutronpy(Data):
    r"""Loads neutronpy format ascii or hdf5 data file into a Data object

    """

    def __init__(self):
        super(Neutronpy, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads data from ascii or hdf5 format file.

        Parameters
        ----------
        filename : str
            Path to file to load

        build_hkl : bool, optional
            Option to build Q = [h, k, l, e, temp]

        load_instrument : bool, optional
            Option to build Instrument from file header

        """
        if filename[-2:].lower() == 'h5':
            self.load_hdf5(filename, build_hkl, load_instrument)
        else:
            with open(filename, 'r') as f:
                first_line = f.readline()
                if 'NeutronPy' in first_line:
                    self.load_ascii(filename, build_hkl, load_instrument)
                else:
                    raise IOError

    def load_hdf5(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads data from HDF5 format file

        Parameters
        ----------
        filename : str
            Path to file to load

        build_hkl : bool, optional
            Option to build Q = [h, k, l, e, temp]

        load_instrument : bool, optional
            Option to build Instrument from file header

        """
        import h5py

        data = OrderedDict()

        with h5py.File(filename, 'r') as f:
            data_root = f['data']

            for key, value in data_root.items():
                data[key] = np.copy(value)

            self._data = data
            self.data_keys = dict((key, str(value)) for key, value in data_root['data_keys'].attrs.items())

            if build_hkl:
                try:
                    self.Q_keys = dict((key, str(value)) for key, value in data_root['Q_keys'].attrs.items())
                except KeyError:
                    warnings.warn('Q_keys could not be built automatically.')

        if load_instrument:
            self.instrument = load_instr(filename, filetype='hdf5')

    def load_ascii(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads data from ascii format file

        Parameters
        ----------
        filename : str
            Path to file to load

        build_hkl : bool, optional
            Option to build Q = [h, k, l, e, temp]

        load_instrument : bool, optional
            Option to build Instrument from file header

        """
        file_header = []
        with open(filename) as f:
            for line in f:
                if '#' in line and 'npy_col_headers' not in line:
                    file_header.append(line.replace('\n', '').replace('# ', ''))
                if 'npy_col_headers' in line:
                    args = next(f).split()
                    col_headers = [head for head in args[1:]]

        args = np.genfromtxt(filename, unpack=True, comments='#', dtype=np.float64)

        data = OrderedDict()
        for head, col in zip(col_headers, args):
            data[head] = col

        self._data = data

        for n, line in enumerate(file_header):
            if 'original_header' in line:
                start = n + 1

            if build_hkl:
                if 'Q_keys' in line:
                    self.Q_keys = eval(line.replace('Q_keys =', ''))

            if 'data_keys' in line:
                self.data_keys = eval(line.replace('data_keys = ', ''))

            if 'def_x' in line:
                self.plot_default_x = line.split('=')[-1].strip()

            if 'def_y' in line:
                self.plot_default_y = line.split('=')[-1].strip()

        if build_hkl and not hasattr(self, 'Q_keys'):
            warnings.warn('Q_keys could not be built automatically.')

        self.file_header = file_header[start:-3]

        if load_instrument:
            try:
                self.instrument = load_instr(filename.split('.')[0] + '.instr', filetype='ascii')
            except IOError:
                warnings.warn('Instrument could not be loaded automatically.')
