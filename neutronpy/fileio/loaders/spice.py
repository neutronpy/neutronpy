from collections import OrderedDict
import numpy as np
from ...data import Data
from ...instrument import Instrument


class Spice(Data):
    def __init__(self):
        __metaclass__ = Data
        super(Spice, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        file_header = []
        with open(filename) as f:
            for line in f:
                if '#' in line:
                    file_header.append(line.replace('\n', '').replace('# ', ''))
                if 'col_headers' in line:
                    args = next(f).split()
                    col_headers = [head for head in args[1:]]

        args = np.genfromtxt(filename, unpack=True, comments='#', dtype=np.float64)

        data = OrderedDict()
        for head, col in zip(col_headers, args):
            data[head] = col

        # delete Pt. column (unnecessary, messes up sorting)
        del data['Pt.']

        self._data = data
        self._file_header = file_header
        self.data_keys = {'monitor': 'monitor', 'detector': 'detector', 'time': 'time'}

        if build_hkl:
            self.Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'tvti'}

        if load_instrument:
            instrument = Instrument()
            for item in file_header:
                key, value = item.split('=')
                if key == 'monochromator':
                    instrument.mono.tau = value
                if key == 'analyzer':
                    instrument.ana.tau = value
                if key == 'collimation':
                    hcol = [float(col) for col in value.split('-')]
                    instrument.hcol = hcol
                if key == 'samplemosaic':
                    instrument.sample.mosaic = float(value)
                if key == 'latticeconstants':
                    instrument.sample.abc = [float(i) for i in value.split(',')[:3]]
                    instrument.sample.abg = [float(i) for i in value.split(',')[3:]]

            self.instrument = instrument
