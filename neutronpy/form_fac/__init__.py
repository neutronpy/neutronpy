import json
import numpy as np

with open('../database/scat_len.json', 'r') as infile:
    scatLen = json.load(infile)

with open('../database/periodic_table.json', 'r') as infile:
    periodicTable = json.load(infile)


class Atom():
    '''
        Class for adding atoms to the **Material class**, which requires:

        * ion: the name of the Atom, or ion if necessary
        * pos: the position of the Atom in the chosen geometry
        * dpos: deviations from the position pos
        * occupancy: occupancy of the Atom (e.g. if there
            partial occupancy from doping)
        * formUnit: number of formula units in a cell
        * massNorm: normalize to the mass of the atom
    '''
    def __init__(self, ion, pos, dpos, occupancy, Mcell=None):
        self.ion = ion
        self.pos = pos
        self.dpos = dpos
        self.occupancy = occupancy

        if Mcell is not None:
            self.Mcell = Mcell
            self.mass = periodicTable[ion]['mass']
            self.b = (scatLen[ion]['Coh b']
                      * self.occupancy
                      * self.Mcell
                      / np.sqrt(self.mass))
        else:
            self.b = scatLen[ion]['Coh b'] * self.occupancy


# TODO: Add handling of mass normalization
# TODO: Add default behavior for Debye-Waller
class Material():
    '''
        Class for the Material being supplied for the 
        structure factor calculation

        data is a dictionary containing all of the atoms
        and their positions, with (optionally) occupancy 
        and variances in positions (dpos), which may be
        used for Debye-Waller factor. This dictionary has
        the format:

        {
            'Atom': {
                'ion': string
                'pos': [float, float, float],
                'dpos: [float, float, float],
                'occupancy': float
                },
        }
    '''
    def __init__(self, data, formUnit=1.):
        self.muCell = 0.
        for value in data['composition']:
            if 'occupancy' not in value:
                value['occupancy'] = 1.
            self.muCell += periodicTable[value['ion']]['mass'] * value['occupancy']
        self.Mcell = formUnit * self.muCell

        self.atoms = []
        for value in data['composition']:
            if 'dpos' not in value:
                value['dpos'] = np.zeros(3)
            if 'occupancy' not in value:
                value['occupancy'] = 1.
            self.atoms.append(Atom(value['ion'],
                                   value['pos'],
                                   value['dpos'],
                                   value['occupancy'],
                                   self.Mcell))
        if 'unitVectors' in data:
            self.abc = data['unitVectors']
            self.abcs = np.array([2. * np.pi / a for a in self.abc])


# TODO: Add Debye-Waller Factor support
def str_fac(unitCell, formUnit=1., h=None, k=None, l=None):
    '''
        Calculates the Structural Form Factor of a given unitCell,
        number of formula units (formUnit), and h, k, l

        return: float or numpy array
    '''
    material = Material(data=unitCell, formUnit=formUnit)

    # Determines shape of input variables to build FQ = 0 array
    if type(h) is float and type(k) is float and type(l) is float:
        FQ = 0. * 1j
    else:
        # if one dimension is zero,
        # flatten FQ to 2D
        if type(h) is float:
            FQ = np.zeros(k.shape) * 1j
        elif type(k) is float:
            FQ = np.zeros(l.shape) * 1j
        elif type(l) is float:
            FQ = np.zeros(h.shape) * 1j
        else:
            FQ = np.zeros(h.shape) * 1j

    # construct structure factor
    for atom in material.atoms:
        FQ += atom.b * np.exp(1j * 2. * np.pi *
                              (h * (atom.pos[0] + atom.dpos[0]) +
                               k * (atom.pos[1] + atom.dpos[1]) +
                               l * (atom.pos[2] + atom.dpos[2])))

    return FQ


# TODO: Add magnetic form factor
def mag_fac():
    pass


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


if __name__ == "__main__":
    test_str_fac()
