# -*- coding: utf-8 -*-
from .takin import TakinTimeOfFlight, TakinTripleAxis
from .tas_instrument import TripleAxisInstrument
from .tof_instrument import TimeOfFlightInstrument


class Instrument(object):
    r"""An object that represents either a Triple Axis Spectrometer instrument
    or a Time of Flight Instrument configuration, including a sample.

    Parameters
    ----------
    args : arg, optional
        Any valid **positional** arguments for the desired instrument class

    instrument_type : str, optional
        Used to select Triple Axis instrument 'tas' or time of flight
        instrument 'tof'. Default: 'tas'

    engine : str, optional
        Used to select the engine for resolution calculations.
        Default: 'neutronpy' for 'tas' instruments, and 'takin' for 'tof'
        instruments.

    kwargs : kwarg, optional
        Any valid keyword arguments for the desired instrument class

    """

    def __init__(self, *args, **kwargs):
        if 'instrument_type' not in kwargs:
            kwargs['instrument_type'] = 'tas'

        if 'engine' not in kwargs:
            kwargs['engine'] = 'neutronpy'

        self.instrument_type = kwargs['instrument_type']
        self.engine = kwargs['engine']

        if kwargs['instrument_type'] == 'tas':
            if kwargs['engine'] == 'neutronpy':
                self.__class__ = TripleAxisInstrument
                self.__init__(*args, **kwargs)
            elif kwargs['engine'] == 'takin':
                self.__class__ = TakinTripleAxis
                self.__init__(*args, **kwargs)

        elif kwargs['instrument_type'] == 'tof':
            if kwargs['engine'] == 'neutronpy':
                self.__class__ = TimeOfFlightInstrument
                self.__init__(*args, **kwargs)

            elif kwargs['engine'] == 'takin':
                self.__class__ = TakinTimeOfFlight
                self.__init__(*args, **kwargs)

    def __repr__(self):
        return "Instrument('{0}', engine='{1}')".format(self.instrument_type, self.engine)
