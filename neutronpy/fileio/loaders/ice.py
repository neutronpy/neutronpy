import numpy as np
from ...data import Data
from ...instrument import Instrument

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Ice(Data):
    r"""Loads ICE (NCNR) format ascii data file.

    """

    def __init__(self):
        super(Ice, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads the ICE (NCNR) format ascii data file.

        Parameters
        ----------
        filename : str
            Path to file to open

        build_hkl : bool, optional
            Option to build Q = [h, k, l, e, temp]

        load_instrument : bool, optional
            Option to build Instrument from file header

        """
        with open(filename) as f:
            file_header = []
            for line in f:
                if 'Columns' in line:
                    args = line.split()
                    col_headers = [head for head in args[1:]]
                    break

        args = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                             unpack=True, comments="#", dtype=np.float64)

        data = OrderedDict()
        for head, col in zip(col_headers, args):
            data[head] = col

        self._data = data
        self.data_keys = {'detector': 'Detector', 'monitor': 'Monitor', 'time': 'Time'}
        self._file_header = file_header

        if build_hkl:
            self.Q_keys = {'h': 'QX', 'k': 'QY', 'l': 'QZ', 'e': 'E', 'temp': 'Temp'}

        if load_instrument:
            instrument = Instrument()
            self.instrument = instrument
