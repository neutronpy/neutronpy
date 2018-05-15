# -*- coding: utf-8 -*-
r"""A class for handeling a collection of scans

"""
import collections

import matplotlib.colors as mpl_c
import matplotlib.pyplot as plt
import numpy as np


class Scans(object):
    r"""A class for a collection of scans

    Attributes
    ----------
    scans : Ordered dict
        A dictionary of scan objects
    num_scans : int
        The number of scans

    Methods
    -------
    waterfall
    pcolor
    func_col
    min_col
    max_col
    mean_col
    update
    scans_check

    """

    def __init__(self, scans_dict=None):
        if scans_dict is not None:
            if not isinstance(scans_dict, collections.OrderedDict):
                raise RuntimeError(
                    "the input dictionary must be of type OrderedDict")
            self.num_scans = len(scans_dict)
        self.scans = scans_dict

    def update(self, scans_dict):
        r"""Update the scans_dict to include the dictionary scans_dict
        This will update any scans that are already in the class and will
        append those that are not.

        Parameters
        ----------
        scans_dict : dict
              A dictionary of multiple scans to add to the collection.

        """
        if not isinstance(scans_dict, collections.OrderedDict):
            raise RuntimeError(
                "the input dictionary must be of type OrderedDict")
        self.scans.update(scans_dict)
        self.num_scans = len(scans_dict)

    def scans_check(self):
        r"""Check to see if their are scans in the object.

        This should be used for other methods that work on individual scans in
        the collection.

        Raises
        ------
        RuntimeError
             If there are no scans

        """
        if self.scans is None:
            raise RuntimeError('There must be at lest one scan')

    def waterfall(self, x='e', y='detector', label_column='h', offset=5, fmt='b-', legend=False, show_plot=False):
        r"""Create a waterfall plot of all the scans in the collection

        Parameters
        ----------
        x : str, optional
            Name of one of the columns that is in each scan to plot along the
            x-axis. Default: `'e'`.

        y : str, optional
            Name of one of the columns that is in each scan to plot along the
            y-axis. Default: `'detector'`.

        label_column : str, optional
            Name of one of the columns that is in each scan to label each
            separate scan in the plot. Default: `'h'`.

        offset : float, optional
            Offset in y, of each successive curve from the previous one.
            Default: 5.

        fmt : str, optional
            Matplotlib plot format string.  Default: `'b-'`.

        legend : bool, optional
            Flag to plot a legend, where True plots a legend. Default: False.

        """
        self.scans_check()
        fh = plt.figure()
        plt.hold(True)

        for idx, scan_num in enumerate(self.scans.keys()):
            xin = self.scans[scan_num].data[x]
            yin = self.scans[scan_num].data[y]
            avg_label_val = self.scans[scan_num].data[label_column].mean()
            label_str = "%s =%1.3f" % (label_column, avg_label_val)
            plt.plot(xin, yin + offset * idx, fmt, label=label_str)
            plt.xlabel(x)
            plt.ylabel(y)

        if legend:
            plt.legend()

        if show_plot:
            plt.show(block=False)

        return fh

    def mean_col(self, col):
        r"""Take the mean of a given column in every scan of the collection

        Parameters
        ----------
        col : str
           The name of the column for the mean

        Returns
        -------
        array_like
            an array where each element is the average of the column of a specific
            scan in the collection

        """
        return self.func_col(col, np.mean)

    def func_col(self, col, func):
        r""" apply a function to a column an return the value for each scan

        Parameters
        ----------
        col : str
            The name of the column for the mean

        func: function
            The function to apply

        """
        self.scans_check()
        res = np.empty(self.num_scans)

        for idx, scan_key in enumerate(self.scans.keys()):
            res[idx] = func(self.scans[scan_key].data[col])

        return res

    def min_col(self, col):
        r""" find the minimum of a given column in every scan of the collection

        Parameters
        ----------
        col : str
            The name of the column for the means

        Returns
        -------
        array_like
                an array where each element is the minimum value of the column of a specific
                scan in the collection

        """
        return self.func_col(col, np.min)

    def max_col(self, col):
        r""" find the minimum of a given collum in every scan of the collection

        Parameters
        ----------
        col : str
            The name of the column for the mean

        Returns
        -------
        array_like
                an array where each element is the minimum value of the column of a specific
                scan in the collection

        """
        return self.func_col(col, np.max)

    def pcolor(self, x, y, z='detector', clims=None, color_norm='linear', cmap='jet', show_plot=True):
        r"""Create a false colormap for a coloction of scans.

        The y-direction is always what varies between scans.

        Parameters
        ----------
        x : str
            Name of one of the columns that is in each scan to plot along
            the x-axis.

        y : str
            Name of one of the columns that is in each scan to plot along
            the y-axis. This parameter varies between scans, but is
            constant over an individual scan.

        z : str, optional
            Name of one of the data columns that is in each scan to plot
            along the z-axis. Default: `'detector'`.

        clims : array_like, optional
            An array of two floats, where the first is the minimum color
            scale, the second is the maximum of the color scale.  By
            default the maximum and minimum of all data. Default: None

        color_norm : str, optional
            Select a `'log'` or `'linear'` scale. Default: `'linear'`.

        cmap : str, optional
            A matplotlib colormaps.  Default: `'jet'`.

        """
        self.scans_check()
        fh = plt.figure()

        # calculate y spacing
        meany = self.mean_col(col=y)

        # generate an array for bin boundaries of the y axis
        biny = np.zeros(len(meany) + 1)

        # generate the bin boundaries internal to the array
        biny[1:-1] = (meany[:-1] + meany[1:]) / 2

        # generate the first bin boundary to be the same distance from the mean as the second bin boundary
        biny[0] = 2 * meany[0] - biny[1]

        # generate the last bin boundary to be the same distance from the mean as the next to last.
        biny[-1] = 2 * meany[-1] - biny[-2]

        if clims is None:
            # calculate intensity range
            intens_max = 0.
            intens_min = 1.

            for idx, scan_num in enumerate(self.scans.keys()):
                maxz = self.scans[scan_num].data[z].max()
                minz = self.scans[scan_num].data[z].min()

                if maxz > intens_max:
                    intens_max = maxz

                if (minz < intens_min) & (minz > 0):
                    intens_min = minz
        else:
            intens_max = clims[1]
            intens_min = clims[0]

        for idx, scan_num in enumerate(self.scans.keys()):
            meansx = self.scans[scan_num].data[x]
            zvals = self.scans[scan_num].data[z]
            yvals = np.array([biny[idx], meany[idx], biny[idx + 1]])
            xvals = np.zeros(len(meansx) + 1)
            xvals[1:-1] = (meansx[:-1] + meansx[1:]) / 2.
            xvals[0] = 2 * meansx[0] - xvals[1]
            xvals[-1] = 2 * meansx[-1] - xvals[-2]
            xmat = np.vstack((xvals, xvals, xvals))
            ymat = np.tile(yvals, (len(xvals), 1)).T
            zmat = np.vstack((zvals, zvals))

            if color_norm == 'log':
                plt.pcolor(xmat, ymat, zmat, norm=mpl_c.LogNorm(
                    vmin=intens_min, vmax=intens_max), cmap=cmap)
            else:
                plt.pcolor(xmat, ymat, zmat, vmin=intens_min,
                           vmax=intens_max, cmap=cmap)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.colorbar()

        if show_plot:
            plt.show(block=False)

        return fh
