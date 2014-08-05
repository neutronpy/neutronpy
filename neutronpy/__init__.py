
__version__ = '0.1-beta'

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
    pass
