r'''NeutronPy: open source python library for neutron scattering data analysis

'''
from __future__ import absolute_import
from . import constants
from . import instrument
from . import material
from . import structure_factors
from . import functions
from . import models
from . import spurion
from . import lattice
from . import io
from .kmpfit import Fitter
from .energy import Energy
from .data import Data

try:
    from . import gui
except ImportError:
    print('Warning: PyQt5 not found, cannot run Resolution GUI')

__version__ = '1.0.0b1'
