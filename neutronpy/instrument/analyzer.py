# -*- coding: utf-8 -*-
from .monochromator import Monochromator


class Analyzer(Monochromator):
    u'''Class containing analyzer information.

    Parameters
    ----------
    tau : float or str
        Tau value for the analyzer

    mosaic : int
        Mosaic of the analyzer in arc minutes

    direct : ±1, optional
        Direction of the analyzer (left or right, -1 or +1, respectively).
        Default: -1 (left-handed coordinate frame).

    vmosaic : int, optional
        Vertical mosaic of the analyzer in arc minutes. Default: None

    height : float, optional
        Height of the analyzer in cm. Default: None

    width : float, optional
        Width of the analyzer in cm. Default: None

    depth : float, optional
        Depth of the analyzer in cm. Default: None

    horifoc : int, optional
        Set to 1 if horizontally focusing analyzer is used. Default: -1

    thickness : float, optional
        Thickness of Analyzer crystal in cm. Required for analyzer
        reflectivity calculation. Default: None

    Q : float, optional
        Kinematic reflectivity coefficient. Required for analyzer
        reflectivity calculation. Default: None

    Returns
    -------
    Analyzer : object

    Attributes
    ----------
    tau
    mosaic
    vmosaic
    direct
    height
    width
    depth
    rh
    rv
    thickness
    horifoc
    Q
    d

    '''
    def __init__(self, tau, mosaic, direct=-1, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None, horifoc=-1, thickness=None, Q=None):
        super(Analyzer, self).__init__(tau, mosaic, direct=-1, vmosaic=None, height=None, width=None, depth=None, rh=None, rv=None)
        if thickness is not None:
            self.thickness = thickness
        if Q is not None:
            self.Q = Q
        self.horifoc = horifoc
