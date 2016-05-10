import numpy as np
from ...data import Data
from ...instrument import Instrument

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Spice(Data):
    r"""Loads SPICE (HFIR) format ascii data file into a Data object

    """

    def __init__(self):
        __metaclass__ = Data
        super(Spice, self).__init__()

    def load(self, filename, build_hkl=True, load_instrument=False):
        r"""Loads the SPICE (HFIR) format ascii data file

        Parameters
        ----------
        filename : str
            Path to file to open

        build_hkl : bool, optional
            Option to build Q = [h, k, l, e, temp]

        load_instrument : bool, optional
            Option to build Instrument from file header

        """
        file_header = []
        with open(filename) as f:
            for line in f:
                if '#' in line and 'col_headers' not in line:
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
        self.file_header = file_header
        self.data_keys = {'monitor': 'monitor', 'detector': 'detector', 'time': 'time'}

        for line in file_header:
            if 'def_x' in line:
                self.plot_default_x = line.split('=')[-1].strip()
            if 'def_y' in line:
                self.plot_default_y = line.split('=')[-1].strip()

        if build_hkl:
            self.Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'tvti'}

        if load_instrument:
            instrument = Instrument()
            for item in file_header:
                line = item.split('=')
                if line[0].strip() == 'monochromator':
                    instrument.mono.tau = line[-1].strip()
                if line[0].strip() == 'analyzer':
                    instrument.ana.tau = line[-1].strip()
                if line[0].strip() == 'collimation':
                    hcol = [float(col.strip()) for col in line[-1].split('-')]
                    instrument.hcol = hcol
                if line[0].strip() == 'samplemosaic':
                    instrument.sample.mosaic = float(line[-1].strip())
                if line[0].strip() == 'latticeconstants':
                    instrument.sample.abc = [float(i.strip()) for i in line[-1].split(',')[:3]]
                    instrument.sample.abg = [float(i.strip()) for i in line[-1].split(',')[3:]]

            self.instrument = instrument
