r'''NeutronPy: open source python library for neutron scattering data analysis

'''
from __future__ import absolute_import

__version__ = '0.3.0'

from . import constants
from . import core as tools
from . import resolution
from . import form_facs
from . import functions
from . import models
from .core import *  # pylint: disable=wildcard-import

try:
    from .kmpfit import Fitter
except ImportError:
    print(ImportError(u'Fitter module is not loaded.'))
