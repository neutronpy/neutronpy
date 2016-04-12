from collections import OrderedDict
import re
import numpy as np
from ...data import Data
from ...instrument import Instrument


class Icp(Data):
    def __init__(self):
        super(Icp, self).__init__()

    def load(self, filename, build_Q=True, load_instrument=False):
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

        if build_Q:
            self.Q_keys = {'h': 'Q(x)', 'k': 'Q(y)', 'l': 'Q(z)', 'e': 'E', 'temp': 'T-act'}

        if load_instrument:
            instrument = Instrument()
            self.instrument = instrument
