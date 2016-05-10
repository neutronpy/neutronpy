r"""NeutronPy: open source python library for neutron scattering data analysis

"""
from __future__ import absolute_import
import warnings
from . import constants
from . import fileio
from . import functions
from . import instrument
from . import models
from . import spurion
from . import scattering
from .crystal import Lattice
from .crystal import Material
from .crystal import Sample
from .crystal import symmetry
from .data import Data
from .energy import Energy
from .kmpfit import Fitter
from .instrument import Instrument

try:
    from . import gui
except ImportError:
    warnings.warn('PyQt5 not found, cannot run Resolution GUI')

__version__ = '1.0.0'
