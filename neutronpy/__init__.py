r'''NeutronPy: open source python library for neutron scattering data analysis

'''
from __future__ import absolute_import
from . import constants
from . import functions
from . import models
from . import spurion
from . import io
from . import polarization
from . import symmetry
from .kmpfit import Fitter
from .energy import Energy
from .data import Data
from . import instrument
from .instrument import Instrument
from .sample import Sample
from .material import Material
from .lattice import Lattice

try:
    from . import gui
except ImportError:
    print('Warning: PyQt5 not found, cannot run Resolution GUI')

__version__ = '1.0.0b1'
