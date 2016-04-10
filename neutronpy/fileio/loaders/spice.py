from collections import OrderedDict
import numpy as np
from neutronpy.data import Data, RawData
from neutronpy.instrument import Instrument


def load(filename, build_hkl=True, load_instrument=False):
    top_header = []
    with open(filename) as f:
        for line in f:
            top_header.append(line)
            if 'col_headers' in line:
                args = next(f).split()
                col_headers = [head for head in args[1:]]
            # grab footer here

    args = np.genfromtxt(filename, unpack=True, comments='#', dtype=np.float64)

    _data = OrderedDict()
    for head, col in zip(col_headers, args):
        _data[head] = col

    for key, value in _data.items():
        print(key, value.shape)

    _t0 = 60.

    if build_hkl:
        data_keys = {'monitor': 'monitor', 'detector': 'detector', 'time': 'time'}
        Q_keys = {'h': 'h', 'k': 'k', 'l': 'l', 'e': 'e', 'temp': 'tvti'}
        raw_data = {}

        if load_instrument:
            return Data(), Instrument()
        else:
            return Data()
    else:
        if load_instrument:
            return RawData(), Instrument()
        else:
            return RawData()


if __name__ == "__main__":
    load('../../../tests/scan0001.dat')
