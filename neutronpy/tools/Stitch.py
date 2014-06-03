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
    # Initialize the output dictionary with empty arrays
    output = {'q': np.array([]), 'inten': np.array([]),
              'err': np.array([]), 'mon': np.array([])}

    '''For each data set passed to the method this loop first checks if there
    is any overlap with the 'q' already in output. If there is no overlap,
    the next data set is just appended to the output arrays. For points that
    overlap, they are added to the output arrays at positions of overlap.'''
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
    '''Combine two or more sets of data (overlapping or non-overlapping)
    *args
        data = {'q': ([],[]), 'inten': [], 'err': [], 'mon': []}
    data['q'] is a tuple of two 1D arrays

    Does not yet work with 2D meshgrid type of arrays
    '''
    output = {'q': (np.array([]), np.array([])), 'inten': np.array([]),
              'err': np.array([]), 'mon': np.array([])}

    for arg in args:
        for key in arg:
            if key != 'q':
                if len(arg[key].shape) > 1:
                    arg[key] = arg[key].flatten()
            else:
                for n in range(len(arg['q'])):
                    if len(arg[key].shape) > 1:
                        arg[key][n] = arg[key][n].flatten()

        if np.any(np.in1d(output['q'][0], arg['q'][0])):
            _temp2combine = np.intersect1d(output['q'][0], arg['q'][0])
            _q2combine = []
            for t in _temp2combine:
                if output['q'][1][np.where(output['q'][0] == t)] == arg['q'][1][np.where(arg['q'][0] == t)]:
                    _q2combine.append(output['q'][0][np.where(output['q'][0] == t)])
            if len(_q2combine) > 0:
                for q in _q2combine:
                    for key in output:
                        if key != 'q':
                            output[key][np.where(output['q'][0] == q)] += arg[key][np.where(arg['q'][0] == q)]
            else:
                break
        else:
            for key in output:
                if key != 'q':
                    output[key] = np.concatenate((output[key], arg[key]))
                else:
                    for n in range(len(output[key])):
                        output[key][n] = np.concatenate((output[key][n], arg[key][n]))

    sort_order = np.lexsort(output['q'][0], output['q'][1])
    for key in output:
        if key != 'q':
            output[key] = np.array(output[key][sort_order])
        else:
            for n in range(len(output[key])):
                output[key][n] = np.array(output[key][n][sort_order])

    return output


def combine3D(*args):
    pass


if __name__ == '__main__':
    pass
