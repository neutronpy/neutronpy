'''
Created on May 27, 2014

@author: davidfobes
'''

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def slice2D(data=None, datalim=None, infile=None, outfile=None, show=True,
            labels=None, smooth=False, colorbar=False, **kwargs):
    if data is not None:
        inten, err, x, y = data  # @UnusedVariable
    elif infile is not None:
        inten, err, x, y = np.loadtxt(infile, unpack=True)  # @UnusedVariable

    if datalim is not None:
        ilim, xlim, ylim, xstep, ystep = (datalim['ilim'], datalim['xlim'],
                                          datalim['ylim'], datalim['xstep'],
                                          datalim['ystep'])
    else:
        ilim = [np.minimum(inten), np.maximum(inten)]
        xlim = [np.minimum(x), np.maximum(x)]
        ylim = [np.minimum(y), np.maximum(y)]
        xstep = np.ceil((xlim[1] - xlim[0]) / (len(x) - 1))
        ystep = np.ceil((ylim[1] - ylim[0]) / (len(y) - 1))

    xi = np.linspace(xlim[0], xlim[1], (xlim[1] - xlim[0]) / xstep + 1)
    yi = np.linspace(ylim[0], ylim[1], (ylim[1] - ylim[0]) / ystep + 1)
    Xi, Yi = np.meshgrid(xi, yi)

    Inten = griddata((x, y), inten, (Xi, Yi), method='nearest')

    if smooth is not False:
        Inten = ndimage.gaussian_filter(Inten, sigma=smooth, mode='nearest')

    plt.pcolormesh(Xi, Yi, Inten, vmin=ilim[0], vmax=ilim[1], **kwargs)

    if labels is not None:
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
        plt.title(labels['title'])

    if outfile is not None:
        plt.savefig(outfile)

    if show:
        plt.show()


def cut1D(data=None):
    pass

if __name__ == '__main__':
    pass
