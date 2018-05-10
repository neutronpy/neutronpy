# -*- coding: utf-8 -*-
r"""Defines Detector class for use in TimeOfFlightInstrument
"""


class Detector(object):
    r"""Class defining a Time of Flight detector

    Parameters
    ----------
    shape : str
        Shape of the detector. 'cylindrical' or 'spherical'

    width : list
        Horizontal coverage of the detector from sample, where 0 is the
        angle of ki where it would hit the detector, in degrees.

    height : list
        Vertical coverage of the detector from the sample, where 0 is the
        position of ki where it would hit the detector, in degrees.

    radius : float
        Radius of curvature of the detector, i.e. the distance
        from the sample to the detector, in cm.

    hpixels : int
        Angular acceptance of a single detector in the horizontal orientation,
        in arc minutes.

    vpixels : int
        Number of detector pixels in the vertical direction

    tau : float, optional
        Binning of the detector in microseconds.

    thickness : float, optional
        Thickness of the detector in cm.

    orientation : string
        Orientation of the cylinder, 'horizontal' or 'vertical', where the
        radius of curvature rotates around the horizontal or vertical axis,
        respectively. Required for shape == 'cylindrical'.

    dead_angles : array-like, optional
        List where dead angles are entered such that [start, stop], in
        degrees. If more than one range of dead angles, pass list of lists.

    Attributes
    ----------
    shape
    width
    height
    radius
    tau
    thickness
    orientation
    dead_angles

    """

    def __init__(self, shape, width, height, radius, hpixels, vpixels, tau=0.1, thickness=1, orientation=None, dead_angles=None):
        self.shape = shape
        self.width = width
        self.height = height
        self.radius = radius
        self.tau = tau
        self.thickness = thickness
        self.hpixels = hpixels
        self.vpixels = vpixels
        if dead_angles:
            self.dead_angles = dead_angles
        if orientation:
            self.orientation = orientation

    def __repr__(self):
        args = ', '.join([str(getattr(self, key)) for key in ['shape', 'width', 'height', 'radius']])
        kwargs = ', '.join(
            ['{0}={1}'.format(key, getattr(self, key, None)) for key in ['resolution', 'orientation', 'dead_angles']])
        return "Detector({0})".format(', '.join([args, kwargs]))