import json
import numpy as np
import neutronpy.constants as const


def test_str_fac():
    structure = {
                 'name': 'Fe1.1Te',
                 'composition': [
                          {'ion': 'Fe', 'pos': [0.75, 0.25, 0.]},
                          {'ion': 'Fe', 'pos': [1. - 0.75, 1. - 0.25, 0.0]},
                          {'ion': 'Te', 'pos': [0.25, 0.25, 1. - 0.2839]},
                          {'ion': 'Te', 'pos': [1. - 0.25, 1. - 0.25, 1. - (1. - 0.2839)]},
                          {'ion': 'Fe', 'pos': [0.25, 0.25, 0.721], 'occupancy': 0.1},
                          {'ion': 'Fe', 'pos': [1. - 0.25, 1. - 0.25, 1. - 0.721], 'occupancy': 0.1}
                        ],
                 'debye-waller': False,
                 'massNorm': True,
                 'formulaUnits': 2.,
                 'unitVectors': [3.81, 3.81, 6.25]
                }

    print(np.abs(str_fac(structure, h=1., k=1., l=0.)) ** 2)
    # 223075.365633
