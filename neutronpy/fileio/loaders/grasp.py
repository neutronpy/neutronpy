from collections import OrderedDict

import numpy as np

from ...data import Data


class Grasp(Data):
    r"""Loads GRASP exported ascii/HDF5 data files.

    """

    def __init__(self):
        super(Grasp, self).__init__()

    def load(self, filename, **kwargs):
        r"""Loads the GRASP (SANS) exported ascii/HDF5 data files, including
        NXS and DAT

        Parameters
        ----------
        filename : str
            Path to file to open

        """

        if filename[-3:].lower() == 'nxs':
            self.load_nxs(filename, **kwargs)
        else:
            with open(filename) as f:
                first_line = f.readline()
            if filename[-3:].lower() == 'dat' or 'GRASP' in first_line:
                self.load_dat(filename, **kwargs)
            else:
                raise IOError

    def load_nxs(self, filename, **kwargs):
        import h5py

        f = h5py.File(filename)

        intensity = np.squeeze(np.array(f['entry0/data1/intensity1']))
        qx = np.squeeze(np.array(f['entry0/data1/qx1']))
        qy = np.squeeze(np.array(f['entry0/data1/qy1']))
        qangles = np.squeeze(np.array(f['entry0/data1/qangle1']))
        mod_q = np.squeeze(np.array(f['entry0/data1/mod_q1']))
        err_intensity = np.squeeze(np.array(f['entry0/data1/err_intensity1']))

        self._data = OrderedDict(intensity=intensity,
                                 qx=qx,
                                 qy=qy,
                                 qangles=qangles,
                                 mod_q=mod_q,
                                 err_intensity=err_intensity,
                                 monitor=np.ones(intensity.shape),
                                 time=np.ones(intensity.shape))

        self.data_keys = {'detector': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = err_intensity

    def load_dat(self, filename, **kwargs):
        with open(filename) as f:
            file_header = []
            for n, line in enumerate(f):
                if 'I' in line and 'Err_I' in line:
                    col_headers = line.replace('\n', '').split()
                    skip_rows = n + 1
                    break
                file_header.append(line.replace('\n', ''))

        data_cols = np.loadtxt(filename, skiprows=skip_rows, unpack=True)
        data = OrderedDict()
        for key, value in zip(col_headers, data_cols):
            data[key] = value

        data['monitor'] = np.ones(data['I'].shape)
        data['time'] = np.ones(data['I'].shape)

        self._data = data
        self.data_keys = {'detector': 'I', 'monitor': 'monitor', 'time': 'time'}
        self._err = data['Err_I']
