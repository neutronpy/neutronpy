import numpy as np
from ...data import Data
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class DcsMslice(Data):
    r'''Loads DCS_MSLICE (NCNR) exported ascii data files.

    '''
    def __init__(self):
        super(DcsMslice, self).__init__()

    def load(self, filename, **kwargs):
        r'''Loads the DCS_MSLICE (NCNR) exported ascii data files, including
        SPE, IEXY, XYE, and XYIE. NOTE: Will NOT load DAVE format files.
        DAVE files are propietary IDL formatted save files.

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
