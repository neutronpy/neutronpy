import numpy as np

from ..lsfit import Fitter
from ..lsfit.tools import convert_params


class PlotData(object):
    """Class containing data plotting methods

    Methods
    -------
    plot
    plot_line
    plot_contour
    plot_volume

    """

    def plot(self, x=None, y=None, z=None, w=None, show_err=True, to_bin=None,
             plot_options=None, fit_options=None, smooth_options=None,
             output_file='', show_plot=True, **kwargs):
        r"""Plots the data in the class. x and y must at least be specified,
        and z and/or w being specified will produce higher dimensional plots
        (contour and volume, respectively).

        Parameters
        ----------
        x : str, optional
            `data_column` key defining the x-axis.
            Default: :py:attr:`plot_default_x`.

        y : str, optional
            `data_column` key defining the y-axis.
            Default: :py:attr:`plot_default_y`.

        z : str, optional
            `data_column` key defining the z-axis.
            Default: None

        w : str, optional
            `data_column` key defining the w-axis.
            Default: None

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each `data_column` key. See
            documentation for :py:meth:`.Data.bin`. Default: None

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: True

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported. Default: None

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine.
            Default: None

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine. Default: None

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`. Default: None

        kwargs : optional
            Additional plotting keyword arguments passed to the plotting
            function.

        """
        if to_bin is None:
            to_bin = dict()
        if plot_options is None:
            plot_options = dict()
        if fit_options is None:
            fit_options = dict()
        if smooth_options is None:
            smooth_options = dict(sigma=0)

        if x is None:
            try:
                x = self.plot_default_x
            except AttributeError:
                raise

        if y is None:
            try:
                y = self.plot_default_y
            except AttributeError:
                raise

        if w is not None:
            self.plot_volume(x, y, z, w, to_bin, plot_options, smooth_options,
                             output_file, show_plot, **kwargs)

        elif w is None and z is not None:
            self.plot_contour(x, y, z, to_bin, plot_options, smooth_options,
                              output_file, show_plot, **kwargs)

        elif w is None and z is None:
            self.plot_line(x, y, show_err, to_bin, plot_options, fit_options,
                           smooth_options, output_file, show_plot, **kwargs)

    def plot_volume(self, x, y, z, w, to_bin=None, plot_options=None, smooth_options=None, output_file='', show_plot=True, **kwargs):
        r"""Plots a 3D volume of 4D data

        Parameters
        ----------
        x : str
            `data_column` key defining the x-axis.
            Default: :py:attr:`plot_default_x`.

        y : str
            `data_column` key defining the y-axis.
            Default: :py:attr:`plot_default_y`.

        z : str
            `data_column` key defining the z-axis.
            Default: None

        w : str
            `data_column` key defining the w-axis.
            Default: None

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each `data_column` key. See
            documentation for :py:meth:`.Data.bin`. Default: None

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: True

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported. Default: None

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine.
            Default: None

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine. Default: None

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`. Default: None

        kwargs : optional
            Additional plotting keyword arguments passed to the plotting
            function.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError('Matplotlib >= 1.3.0 is necessary for plotting.')

        if to_bin:
            data_bin = self.bin(to_bin)
            _x = data_bin.data[x]
            _y = data_bin.data[y]
            _z = data_bin.data[z]
            if w == 'intensity':
                _w = data_bin.intensity
            else:
                _w = data_bin.data[w]
        else:
            _x = self.data[x]
            _y = self.data[y]
            _z = self.data[z]
            if w == 'intensity':
                _w = self.intensity
            else:
                _w = self.data[w]

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            _w = gaussian_filter(_w, **smooth_options)

        _x, _y, _z, _w = (np.ma.masked_where(_w <= 0, _x),
                          np.ma.masked_where(_w <= 0, _y),
                          np.ma.masked_where(_w <= 0, _z),
                          np.ma.masked_where(_w <= 0, _w))

        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')

        axis.scatter(_x, _y, _z, c=_w, linewidths=0, vmin=1.e-4,
                     vmax=0.1, norm=colors.LogNorm())

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass

    def plot_contour(self, x, y, z, to_bin=None, plot_options=None, smooth_options=None, output_file='', show_plot=True, **kwargs):
        r"""Method for plotting a 2D contour plot of 3D data

        Parameters
        ----------
        x : str
            `data_column` key defining the x-axis.
            Default: :py:attr:`plot_default_x`.

        y : str
            `data_column` key defining the y-axis.
            Default: :py:attr:`plot_default_y`.

        z : str
            `data_column` key defining the z-axis.
            Default: None

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each `data_column` key. See
            documentation for :py:meth:`.Data.bin`. Default: None

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: True

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported. Default: None

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine.
            Default: None

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine. Default: None

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`. Default: None

        kwargs : optional
            Additional plotting keyword arguments passed to the plotting
            function.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise

        if to_bin:
            data_bin = self.bin(to_bin)
            _x = data_bin.data[x]
            _y = data_bin.data[y]
            if z == 'intensity':
                _z = data_bin.intensity
            else:
                _z = data_bin.data[z]
        else:
            _x = self.data[x]
            _y = self.data[y]
            if z == 'intensity':
                _z = self.intensity
            else:
                _z = self.data[z]

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            _z = gaussian_filter(_z, **smooth_options)

        x_step = np.around(
            np.abs(np.unique(_x) - np.roll(np.unique(_x), 1))[1], decimals=4)
        y_step = np.around(
            np.abs(np.unique(_y) - np.roll(np.unique(_y), 1))[1], decimals=4)
        x_sparse = np.linspace(
            _x.min(), _x.max(), (_x.max() - _x.min()) / x_step + 1)
        y_sparse = np.linspace(
            _y.min(), _y.max(), (_y.max() - _y.min()) / y_step + 1)
        X, Y = np.meshgrid(x_sparse, y_sparse)

        from scipy.interpolate import griddata

        Z = griddata((_x, _y), _z, (X, Y))

        plt.pcolormesh(X, Y, Z, **plot_options)

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass

    def plot_line(self, x, y, show_err=True, to_bin=None, plot_options=None, fit_options=None, smooth_options=None, output_file='', show_plot=True, **kwargs):
        r"""Method to Plot a line of 2D data

        Parameters
        ----------
        x : str
            `data_column` key defining the x-axis.
            Default: :py:attr:`plot_default_x`.

        y : str
            `data_column` key defining the y-axis.
            Default: :py:attr:`plot_default_y`.

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each `data_column` key. See
            documentation for :py:meth:`.Data.bin`. Default: None

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: True

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported. Default: None

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine.
            Default: None

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine. Default: None

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`. Default: None

        kwargs : optional
            Additional plotting keyword arguments passed to the plotting
            function.

        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise

        if to_bin:
            data_bin = self.bin(to_bin)
            _x = data_bin.data[x]
            if y == 'intensity':
                _y = data_bin.intensity
                _err = data_bin.error
            else:
                _y = data_bin.data[y]
                _err = np.sqrt(data_bin.data[y])
        else:
            _x = self.data[x]
            if y == 'intensity':
                _y = self.intensity
                _err = self.error
            else:
                _y = self.data[y]
                _err = np.sqrt(self.data[y])

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            _y = gaussian_filter(_y, **smooth_options)

        if not plot_options:
            plot_options['fmt'] = 'rs'

        if show_err:
            plt.errorbar(_x, _y, yerr=_err, **plot_options)
        else:
            plt.errorbar(_x, _y, **plot_options)

        # add axis labels
        plt.xlabel(x)
        plt.ylabel(y)

        if fit_options:
            def residuals(params, data):
                funct, x, y, err = data

                return (y - funct(params, x)) / err

            fitobj = Fitter(residuals, data=(
                fit_options['function'], _x, _y, _err))
            if 'fixp' in fit_options:
                fitobj.parinfo = [{'fixed': fix}
                                  for fix in fit_options['fixp']]
            try:
                fitobj.fit(params0=fit_options['p'])
                fit_x = np.linspace(min(_x), max(_x), len(_x) * 10)
                fit_y = fit_options['function'](fitobj.params, fit_x)
                plt.plot(fit_x, fit_y, '{0}-'.format(plot_options['fmt'][0]))

                param_string = u'\n'.join(['p$_{{{0:d}}}$: {1:.3f}'.format(i, p)
                                           for i, p in enumerate(fitobj.params)])
                chi2_params = u'$\chi^2$: {0:.3f}\n\n'.format(
                    fitobj.chi2_min) + param_string

                plt.annotate(chi2_params, xy=(0.05, 0.95), xycoords='axes fraction',
                             horizontalalignment='left', verticalalignment='top',
                             bbox=dict(alpha=0.75, facecolor='white', edgecolor='none'))

            except Exception as mes:  # pylint: disable=broad-except
                raise Exception("Something wrong with fit: {0}".format(mes))

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass
