# -*- coding: utf-8 -*-
import numpy as np
from .tools import GetTau


class Monochromator(object):
    u'''Class containing monochromator information.

    Parameters
    ----------
    tau : float or str
        Tau value for the monochromator (or analyzer)

    mosaic : int
        Mosaic of the crystal in arc minutes

    direct : ±1, optional
        Direction of the crystal (left or right, -1 or +1, respectively).
        Default: -1 (left-handed coordinate frame).

    vmosaic : int, optional
        Vertical mosaic of the crystal in arc minutes. Default: None

    height : float, optional
        Height of the crystal in cm. Default: None

    width : float, optional
        Width of the crystal in cm. Default: None

    depth : float, optional
        Depth of the crystal in cm. Default: None

    Returns
    -------
    Monochromator : object

    Attributes
    ----------
    tau
    mosaic
    direct
    vmosaic
    height
    width
    depth
    rh
    rv
    d
    '''
    def __init__(self, tau, mosaic, direct=-1, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None):
        self._tau = tau
        self.mosaic = mosaic
        if vmosaic is not None:
            self.vmosaic = vmosaic
        self.dir = direct
        self.d = 2 * np.pi / GetTau(tau)
        if rh is not None:
            self.rh = rh
        if rv is not None:
            self.rv = rv
        if height is not None:
            self.height = height
        if width is not None:
            self.width = width
        if depth is not None:
            self.depth = depth

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self.d = 2 * np.pi / GetTau(tau)