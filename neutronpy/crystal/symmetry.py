# -*- coding: utf-8 -*-
r"""Symmetry operations

"""
import numpy as np

from ..constants import symmetry

space_groups = symmetry()['space_groups']


class SpaceGroup(object):
    r"""Class defining a space group of a crystal

    Attributes
    ----------
    full_name
    generators
    group_number
    hm_symbol
    lattice_type
    string_generators
    symbol
    symmetry_operations
    total_operations

    Methods
    -------
    symmetrize_position

    """
    def __init__(self, symbol='P1'):
        if isinstance(symbol, int):
            for key, value in space_groups.items():
                if value['number'] == symbol:
                    self._symbol = key
        elif isinstance(symbol, str):
            if symbol in space_groups:
                self._symbol = symbol
            else:
                for key, value in space_groups.items():
                    if value['hermann-manguin_symbol'] == symbol or value['full_name'] == symbol:
                        self._symbol = key
        else:
            raise KeyError('{0} is not a valid International symbol, Hermann–Mauguin symbol, or space group number'.format(symbol))

        self.point_group = space_groups[self.symbol]['point_group']
        self.full_name = space_groups[self.symbol]['full_name']
        self._generators_str = space_groups[self.symbol]['generators']
        self.lattice_type = space_groups[self.symbol]['type']
        self.group_number = space_groups[self.symbol]['number']
        self.hm_symbol = space_groups[self.symbol]['hermann-manguin_symbol']
        self._generators_mat = get_generator_from_str(self._generators_str)
        self.total_operations = space_groups[self.symbol]['total_operations']
        self.symmetry_operations = self._symmetry_operations_from_generators()

    def __repr__(self):
        return "SpaceGroup({0})".format(self.group_number)

    @property
    def symbol(self):
        r"""Space group symbol
        """
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self.__init__(symbol)

    @property
    def string_generators(self):
        r"""Space group generators
        """
        return self._generators_str

    @property
    def generators(self):
        r"""Space group generators in matrix format
        """
        return self._generators_mat

    def _symmetry_operations_from_generators(self):
        symm_ops, new_ops = [np.copy(self.generators)] * 2
        while len(new_ops) > 0 and len(symm_ops) < self.total_operations:
            gen_ops = []
            for g in new_ops:
                test_ops = np.einsum('ijk,kl', symm_ops, g)
                for op in test_ops:
                    op[:3, 3] = np.mod(get_translation(op), 1)
                    op[np.where(np.abs(1 - get_translation(op)) < 1e-15), 3] = 0
                    if not (np.abs(symm_ops - op) < 1e-15).all(axis=(1, 2)).any():
                        gen_ops.append(op)
                        symm_ops = np.append(symm_ops, [op], axis=0)
            new_ops = gen_ops
        assert len(symm_ops) == self.total_operations
        return symm_ops

    def symmetrize_position(self, vector):
        r"""Applies symmetry operations to a vector

        """
        positions = []
        for op in self.symmetry_operations:
            positions.append(np.dot(get_rotation(op), np.array(vector)) + get_translation(op))

        return positions


def get_formatted_operations(operations):
    r"""Returns operations formatted in a list for easy parsing

    Parameters
    ----------
    operations

    Returns
    -------

    """
    if (isinstance(operations, list) and isinstance(operations[0], str)) or isinstance(operations, str):
        operations = get_generator_from_str(operations)

    if isinstance(operations, np.ndarray):
        operations = [operations]

    return operations


def get_rotation(operations):
    r"""Returns rotational part of operator

    """
    rotations = []
    for operation in get_formatted_operations(operations):
        rotations.append(operation[:3, :3])

    if len(rotations) == 1:
        rotations = rotations[0]

    return rotations


def get_translation(operations):
    r"""Returns rotational part of operator

    """
    translations = []
    for operation in get_formatted_operations(operations):
        translations.append(operation[:3, 3])

    if len(translations) == 1:
        translations = translations[0]

    return translations


def get_generator_from_str(operations):
    r"""Returns generator arrays

    Returns
    -------
    operators : list of ndarrays
        List of operation arrays with shape (3,4)

    """
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

        out = np.zeros((4, 4), dtype=float)
        out[0:3, 0:3] = rotation
        out[0:3, 3] = translation
        out[3, 3] = 1.

        operators.append(out)

    if len(operators) == 1:
        operators = operators

    return operators


def get_str_from_generator(operations):
    r"""Returns strings of generators from arrays

    Parameters
    ----------
    operations : str, array, list

    Returns
    -------
    generators : list of str
        List of generator strings

    """
    if isinstance(operations, np.ndarray) and len(operations.shape) < 3:
        operations = [operations]

    syms = ['x', 'y', 'z']
    signs = {-1: '-', 1: '+', 0: ''}
    generators = []
    for operation in operations:
        line = []
        for row in operation[:3, :]:
            element = ''
            for col, sym in zip(row[:3], syms):
                element += signs[int(col)] + np.abs(int(col)) * sym

            if row[3] == 0:
                translate = ''
            elif np.round(1. / row[3], 1) == 1.5:
                translate = '2/3'
            elif row[3] == 0.75:
                translate = '3/4'
            elif np.round(row[3], 3) == 0.833:
                translate = '5/6'
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
