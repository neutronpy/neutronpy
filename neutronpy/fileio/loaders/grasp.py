import numpy as np
from ...data import Data
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class Grasp(Data):
    r'''Loads GRASP exported ascii/HDF5 data files.

    '''
    def __init__(self):
        super(Grasp, self).__init__()

    def load(self, filename, **kwargs):
        r'''Loads the GRASP (SANS) exported ascii/HDF5 data files, including
        NXS and DAT

        Parameters
        ----------
        filename : str
            Path to file to open

        '''
        pass

    def determine_subtype(self, filename):
        with open(file) as f:
            first_line = f.readline()
            if filename[-3:].lower() == 'nxs' or 'HDF' in first_line:
                return 'nxs'
            if filename[-3:].lower() == 'dat' or 'GRASP' in first_line:
                return 'dat'

    def load_nxs(self, filename):
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

        self.data_keys = {'intensity': 'intensity', 'monitor': 'monitor', 'time': 'time'}
        self._err = err_intensity

    def load_dat(self):
        pass
