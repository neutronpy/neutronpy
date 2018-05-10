# -*- coding: utf-8 -*-
r"""Chopper class for Time of Flight instrument

"""
import numpy as np

from .exceptions import ChopperError


class Chopper(object):
    r"""

    Parameters
    ----------
    distance : float
        Distance of the chopper from the source in cm.

    speed : float
        Speed of the chopper in Hz.

    width : float
        width of the beam at the chopper in cm.

    chopper_type : string
        The type of chopper: 'disk' or 'fermi'.

    acceptance : float
        If chopper_type == 'disk', angular acceptance of the chopper in
        degrees, unless `radius` is defined, in which case `acceptance`
        is the size of the opening in the disk in cm. If chopper_type ==
        'fermi', distance between chopper blades in cm.

    counter_rot : bool, optional
        If the disk chopper consists of two counter rotating choppers, set to
        True (Default: False).

    radius : float, optional
        radius of the chopper in cm. If defined, and chopper_type == 'disk',
        then `acceptance` is assumed to be in units of cm.

    depth : float, optional
        The depth of the Fermi Chopper blades, to calculate angular
        acceptance. Required if chopper_type == 'fermi'.

    tau : float, optional
        Custom value of the resolution of the chopper in standard deviation in
        units of microseconds. Used to override the automatic calculation of
        tau.

    Attributes
    ----------
    distance
    speed
    width
    chopper_type
    acceptance
    radius
    depth
    tau

    """
    def __init__(self, distance, speed, width, chopper_type, acceptance, counter_rot=False, radius=None, depth=None, tau=None):
        self.distance = distance
        self.speed = speed
        self.width = width
        self.chopper_type = chopper_type
        self.acceptance = acceptance
        if counter_rot:
            self.counter_rot = 2.0
        else:
            self.counter_rot = 1.0
        if radius is not None:
            self.radius = radius
        if depth is not None:
            self.depth = depth
        if tau is not None:
            self.tau_override = tau


    def __repr__(self):
        args = ', '.join(
            [str(getattr(self, key)) for key in ['distance', 'speed', 'width', 'chopper_type', 'acceptance']])
        kwargs = ', '.join(
            ['{0}={1}'.format(getattr(self, key)) for key in ['depth', 'tau'] if getattr(self, key, None) is not None])
        return "Chopper({0})".format(', '.join([args, kwargs]))

    @property
    def tau(self):
        """Calculate the time resolution of the chopper

        Returns
        -------
        tau : float
            Returns the resolution of the chopper in standard deviation in units of microseconds
        """
        if hasattr(self, 'tau_override'):
            return self.tau_override

        elif self.chopper_type == 'disk' and hasattr(self, 'radius'):
            return self.acceptance / (self.radius * self.speed * self.counter_rot) / np.sqrt(8 * np.log(2))
        elif self.chopper_type == 'disk' and ~hasattr(self, 'radius'):
            return 1e6 / (self.speed * self.acceptance * self.counter_rot) / 360.0
        elif self.chopper_type == 'fermi':
            try:
                return 1e6 / (self.speed * 2.0 * np.arctan(self.acceptance / self.depth)) / 360.
            except AttributeError:
                raise ChopperError("'depth' not specified, and is a required value for a Fermi Chopper.")

        else:
            raise ChopperError("'{0}' is an invalid chopper_type. Choose 'disk' or 'fermi', or specify custom tau \
                                via `tau_override` attribute".format('chopper_type'))
