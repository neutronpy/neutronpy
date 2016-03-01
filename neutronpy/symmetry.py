# -*- coding: utf-8 -*-
r'''Symmetry operations

'''
import copy
import numpy as np
from neutronpy.constants import symmetry

space_groups = symmetry()['space_groups']
point_groups = symmetry()['point_groups']
lattice_translations = symmetry()['lattice_translations']
wyckoff_info = symmetry()['wyckoff_info']
space_group_info = symmetry()['space_group_info']
Latt_vec = lattice_translations['vectors']


class SpaceGroup(object):
    r'''Space group of a crystal

    '''
    def __init__(self, symbol='P1'):
        if isinstance(symbol, int):
            for key, value in space_groups.items():
                if value['group'] == symbol:
                    self._symbol = key
        elif isinstance(symbol, str):
            self._symbol = symbol.replace(' ', '')
        try:
            self._generator_str = space_groups[self.symbol]['gen']
            self.lattice_type = space_groups[self.symbol]['type']
            self.group_number = space_groups[self.symbol]['group']
            self._generator_mat = get_generator_from_str(self._generator_str)
        except KeyError:
            raise KeyError('{0} is not a valid Hermann–Mauguin symbol'.format(self._symbol))

    @property
    def symbol(self):
        r'''Space group symbol
        '''
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self.__init__(symbol)

    @property
    def generators(self):
        r'''Space group generators
        '''
        return self._generator_mat

    @generators.setter
    def generators(self, operation):
        self._generator_mat = operation

    @property
    def generators_str(self):
        r'''Space group generators in string format
        '''
        return self._generator_str

    @property
    def symmetry_operations(self):
        r'''Symmetry operators given by generators
        '''
        return self._get_symmetry_operations()


def apply_symmetry_operation(operation, vector):
    if not isinstance(operation, np.ndarray):
        operation = get_generator_from_str(operation)

    return get_rotation(operation) * np.array(vector).reshape((3, 1)) + get_translation(operation)


def get_rotation(operations):
    r'''Returns rotational part of operator

    '''
    if (isinstance(operations, list) and isinstance(operations[0], str)) or isinstance(operations, str):
        operations = get_generator_from_str(operations)

    if isinstance(operations, np.ndarray):
        operations = [operations]

    rotations = []
    for operation in operations:
        rotations.append(operation[:, :3])

    if len(rotations) == 1:
        rotations = rotations[0]

    return rotations


def get_rotational_order(operations):
    r'''Returns order of rotation operator(s)

    '''
    if isinstance(operations, list) and isinstance(operations[0], np.ndarray) and np.all(np.unique([i.shape for i in operations]) == [3]):
        pass
    else:
        operations = get_rotation(operations)

    if isinstance(operations, np.ndarray):
        operations = [operations]

    order = []
    for operation in operations:
        det = np.linalg.det(operation)
        tr = operation.trace()

        if tr == -3 and det == -1:
            order.append(-1)
        elif tr == -2 and det == -1:
            order.append(-6)
        elif tr == -1 and det == -1:
            order.append(-4)
        elif tr == -1 and det == 1:
            order.append(2)
        elif tr == 0 and det == -1:
            order.append(-3)
        elif tr == 0 and det == 1:
            order.append(3)
        elif tr == 1 and det == -1:
            order.append(-2)
        elif tr == 1 and det == 1:
            order.append(4)
        elif tr == 2 and det == 1:
            order.append(6)
        elif tr == 3 and det == 1:
            order.append(1)
        else:
            order.append(0)

    if len(order) == 1:
        order = order[0]

    return order


def get_translation(operations):
    r'''Returns rotational part of operator

    '''
    if (isinstance(operations, list) and isinstance(operations[0], str)) or isinstance(operations, str):
        operations = get_generator_from_str(operations)

    if isinstance(operations, np.ndarray):
        operations = [operations]

    translations = []
    for operation in operations:
        translations.append(operation[:, 3])

    if len(translations) == 1:
        translations = translations[0]

    return translations


def get_generator_from_str(operations):
    r'''Returns generator arrays

    Returns
    -------
    operators : list of ndarrays
        List of operation arrays with shape (3,4)

    '''
    if isinstance(operations, str):
        operations = [operations]

    operators = []
    for gen in operations:
        components = gen.split(',')
        if len(components) > 3:
            raise ValueError('Generator string {0} is in wrong format'.format(gen))

        rotation = np.zeros((3, 3))
        translation = np.zeros(3)
        for i, comp in enumerate(components):
            elements = comp.split('+')
            if len(elements) > 1:
                translation[i] = eval(elements[-1].replace('/', './'))

            if '-x' in elements[0]:
                rotation[i, 0] = -1
            elif 'x' in elements[0]:
                rotation[i, 0] = 1

            if '-y' in elements[0]:
                rotation[i, 1] = -1
            elif 'y' in elements[0]:
                rotation[i, 1] = 1

            if '-z' in elements[0]:
                rotation[i, 2] = -1
            elif 'z' in elements[0]:
                rotation[i, 2] = 1

        operators.append(np.hstack((rotation, translation.reshape((3, 1)))))

    if len(operators) == 1:
        operators = operators[0]

    return operators


def get_str_from_generator(operations):
    r'''Returns strings of generators from arrays

    Parameters
    ----------
    operations : str, array, list

    Returns
    -------
    generators : list of str
        List of generator strings

    '''
    if isinstance(operations, np.ndarray):
        operations = [operations]

    syms = ['x', 'y', 'z']
    signs = {-1: '-', 1: '+', 0: ''}
    generators = []
    for operation in operations:
        line = []
        for row in operation:
            element = ''
            for col, sym in zip(row[:3], syms):
                element += signs[col] + np.abs(col) * sym

            if row[3] == 0:
                translate = ''
            elif np.round(1. / row[3], 1) == 1.5:
                translate = '2/3'
            elif row[3] == 0.75:
                translate = '3/4'
            else:
                denominator = int(np.round(1. / row[3]))
                translate = '1/{0}'.format(denominator)

            if translate != '':
                element += '+{0}'.format(translate)

            if len(element) >= 1 and element[0] == '+':
                element = element[1:]

            line.append(element)

        generators.append(','.join(line))

    return generators



