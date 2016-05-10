import numpy as np
from ...data import Data
from ...instrument import Instrument

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Mad(Data):
    r"""Loads MAD (ILL) format ascii data file into a Data object

    """

    def __init__(self):
        super(Mad, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads the MAD (ILL) format ascii data file.

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
            for i, line in enumerate(f):
                if 'DATA_:' in line:
                    args = next(f).split()
                    col_headers = [head for head in args]
                    skip_lines = i + 2
                    break

        args = np.genfromtxt(filename, unpack=True, dtype=np.float64, skip_header=skip_lines, skip_footer=1)

        data = OrderedDict()
        for head, col in zip(col_headers, args):
            data[head] = col

        self._data = data
        self._file_header = file_header
        self.data_keys = {'detector': 'CNTS', 'time': 'TIME', 'monitor': 'M1'}

        if build_hkl:
            self.Q_keys = {'h': 'QH', 'k': 'QK', 'l': 'QL', 'e': 'EN', 'temp': 'TT'}

        if load_instrument:
            instrument = Instrument()
            self.instrument = instrument
