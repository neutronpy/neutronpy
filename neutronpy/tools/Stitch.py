'''
Created on May 28, 2014

@author: davidfobes
'''

import numpy as np


def combine1D(*args):
    '''Combine two or more sets of data (overlapping or non-overlapping)
    *args
        data = {'q': [], 'inten': [], 'err': [], 'mon': []}
    '''
    output = {'q': np.array([]), 'inten': np.array([]),
              'err': np.array([]), 'mon': np.array([])}

    for arg in args:
        if np.any(np.in1d(output['q'], arg['q'])):
            _q2combine = np.intersect1d(output['q'], arg['q'])
            for q in _q2combine:
                for key in output:
                    if key != 'q':
                        output[key][np.where(output['q'] == q)] += arg[key][np.where(arg['q'] == q)]  # @IgnorePep8
            for q in arg['q']:
                if q not in _q2combine:
                    for key in output:
                        if key != 'q':
                            output[key] = np.concatenate((output[key], arg[key][np.where(arg['q'] == q)]))  # @IgnorePep8
                        else:
                            output[key] = np.append(output[key], q)
        else:
            for key in output:
                output[key] = np.concatenate((output[key], arg[key]))

    sort_order = np.argsort(output['q'])
    for key in output:
        output[key] = np.array(output[key][sort_order])

    return output


def combine2D(*args):
    pass


def combine3D(*args):
    pass


if __name__ == '__main__':
    pass
