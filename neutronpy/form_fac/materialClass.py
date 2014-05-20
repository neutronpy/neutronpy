'''
Created on May 20, 2014

@author: davidfobes
'''


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
            self.mass = const.periodicTable[ion]['mass']
            self.b = (const.scatLen[ion]['Coh b']
                      * self.occupancy
                      * self.Mcell
                      / np.sqrt(self.mass))
        else:
            self.b = const.scatLen[ion]['Coh b'] * self.occupancy


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
            self.muCell += const.periodicTable[value['ion']]['mass'] * value['occupancy']
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