import re
import numpy as np
from ...data import Data
from ...instrument import Instrument

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Icp(Data):
    r"""Loads ICP (NCNR) format ascii data file into a Data object

    """

    def __init__(self):
        super(Icp, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads the ICP (NCNR) format ascii data file.

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
                if i == 0:
                    _length = int(re.findall(r"(?='(.*?)')", line)[-2])
                    [_m0, _prf] = [float(i) for i in re.findall(r"(?='(.*?)')", line)[-4].split()]
                if 'Q(x)' in line:
                    args = line.split()
                    col_headers = [head for head in args]
                    break

        args = np.genfromtxt(filename, unpack=True, dtype=np.float64, skip_header=12)

        data = OrderedDict()
        for head, col in zip(col_headers, args):
            data[head] = col

        # add monitor column from header values
        data['monitor'] = np.empty(args[0].shape)
        data['monitor'].fill(_m0 * _prf)

        self._data = data
        self._file_header = file_header
        self.data_keys = {'detector': 'Counts', 'time': 'min', 'monitor': 'monitor'}

        if build_hkl:
            self.Q_keys = {'h': 'Q(x)', 'k': 'Q(y)', 'l': 'Q(z)', 'e': 'E', 'temp': 'T-act'}

        if load_instrument:
            instrument = Instrument()
            self.instrument = instrument
