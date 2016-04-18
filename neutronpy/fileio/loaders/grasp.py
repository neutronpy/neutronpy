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

    def determine_subtype(self):
        pass

    def load_iexy(self):
        pass

    def load_spe(self):
        pass

    def load_xyie(self):
        pass

    def load_xye(self):
        pass
