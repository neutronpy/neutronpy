from __future__ import absolute_import

__version__ = '0.1'

from . import constants
from . import tools
from . import resolution
from . import form_facs
from . import functions
from . import models

try:
    from .kmpfit import Fitter
except ImportError:
    print(ImportError(u'Fitter module is not loaded.'))
