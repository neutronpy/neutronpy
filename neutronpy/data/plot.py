import numpy as np
from ..kmpfit import Fitter


class PlotData(object):
    '''Class containing data plotting methods

    Methods
    -------
    plot

    '''
    def plot(self, x, y, z=None, w=None, show_err=True, to_bin=None,
             plot_options=None, fit_options=None, smooth_options=None,
             output_file='', show_plot=True, **kwargs):
        r'''Plots the data in the class. x and y must at least be specified,
        and z and/or w being specified will produce higher dimensional plots
        (contour and volume, respectively).

        Parameters
        ----------
        x : str
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        y : str
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        z : str, optional
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        w : str, optional
            String indicating the content of the dimension: 'h', 'k', 'l',
            'e', 'temp', or 'intensity'

        bounds : dict, optional
            If set, data will be rebinned to the specified parameters, in the
            format `[min, max, num points]` for each 'h', 'k', 'l', 'e',
            and 'temp'

        show_err : bool, optional
            Plot error bars. Only applies to xy scatter plots. Default: True

        show_plot : bool, optional
            Execute `plt.show()` to show the plot. Incompatible with
            `output_file` param. Default: True

        output_file : str, optional
            If set, the plot will be saved to the location given, in the format
            specified, provided that the format is supported.

        plot_options : dict, optional
            Plot options to be passed to the the matplotlib plotting routine

        fit_options : dict, optional
            Fitting options to be passed to the Fitter routine

        smooth_otions : dict, optional
            Smoothing options for Gaussian smoothing from
            `scipy.ndimage.filters.gaussian_filter`

        Returns
        -------
        None

        '''
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors  # @UnusedImport
        except ImportError:
            ImportError('Matplotlib >= 1.3.0 is necessary for plotting.')

        if to_bin is None:
            to_bin = {}
        if plot_options is None:
            plot_options = {}
        if fit_options is None:
            fit_options = {}
        if smooth_options is None:
            smooth_options = {'sigma': 0}

        args = {'x': x, 'y': y, 'z': z, 'w': w}
        options = ['h', 'k', 'l', 'e', 'temp', 'intensity']

        in_axes = np.array([''] * len(options))
        for key, value in args.items():
            if value is not None:
                in_axes[np.where(np.array(options) == value[0])] = key

        if to_bin:
            binned_data = self.bin(to_bin)
            to_plot = np.where(binned_data.monitor > 0)
            dims = {'h': binned_data.h[to_plot],
                    'k': binned_data.k[to_plot],
                    'l': binned_data.l[to_plot],
                    'e': binned_data.e[to_plot],
                    'temp': binned_data.temp[to_plot],
                    'intensity': binned_data.intensity[to_plot],
                    'error': binned_data.error[to_plot]}
        else:
            to_plot = np.where(self.monitor > 0)
            dims = {'h': self.h[to_plot],
                    'k': self.k[to_plot],
                    'l': self.l[to_plot],
                    'e': self.e[to_plot],
                    'temp': self.temp[to_plot],
                    'intensity': self.intensity[to_plot],
                    'error': self.error[to_plot]}

        if smooth_options['sigma'] > 0:
            from scipy.ndimage.filters import gaussian_filter
            dims['intensity'] = gaussian_filter(dims['intensity'],
                                                **smooth_options)

        x = dims[args['x']]
        y = dims[args['y']]

        if z is not None and w is not None:
            try:
                z = dims[args['z']]
                w = dims[args['w']]

                x, y, z, w = (np.ma.masked_where(w <= 0, x),
                              np.ma.masked_where(w <= 0, y),
                              np.ma.masked_where(w <= 0, z),
                              np.ma.masked_where(w <= 0, w))
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                except ImportError:
                    raise ImportError('Matplotlib >= 1.3.0 is necessary for plotting.')

                fig = plt.figure()
                axis = fig.add_subplot(111, projection='3d')

                axis.scatter(x, y, z, c=w, linewidths=0, vmin=1.e-4,
                             vmax=0.1, norm=colors.LogNorm())

            except KeyError:
                raise

        elif z is not None and w is None:
            try:
                z = dims[args['z']]

                x_step = np.around(np.abs(np.unique(x) - np.roll(np.unique(x), 1))[1], decimals=4)
                y_step = np.around(np.abs(np.unique(y) - np.roll(np.unique(y), 1))[1], decimals=4)
                x_sparse = np.linspace(x.min(), x.max(), (x.max() - x.min()) / x_step + 1)
                y_sparse = np.linspace(y.min(), y.max(), (y.max() - y.min()) / y_step + 1)
                X, Y = np.meshgrid(x_sparse, y_sparse)

                from scipy.interpolate import griddata

                Z = griddata((x, y), z, (X, Y))

                plt.pcolormesh(X, Y, Z, **plot_options)
            except (KeyError, ImportError):
                raise
        else:
            if not plot_options:
                plot_options['fmt'] = 'rs'

            if show_err:
                err = np.sqrt(dims['intensity'])
                plt.errorbar(x, y, yerr=err, **plot_options)
            else:
                plt.errorbar(x, y, **plot_options)

            # add axis labels
            plt.xlabel(args['x'])
            plt.ylabel(args['y'])
            if fit_options:
                def residuals(params, data):
                    funct, x, y, err = data

                    return (y - funct(params, x)) / err

                fitobj = Fitter(residuals, data=(fit_options['function'], x, y,
                                                 np.sqrt(dims['intensity'])))
                if 'fixp' in fit_options:
                    fitobj.parinfo = [{'fixed': fix} for fix in
                                      fit_options['fixp']]

                try:
                    fitobj.fit(params0=fit_options['p'])
                    fit_x = np.linspace(min(x), max(x), len(x) * 10)
                    fit_y = fit_options['function'](fitobj.params, fit_x)
                    plt.plot(fit_x, fit_y, '{0}-'.format(plot_options['fmt'][0]))

                    param_string = u'\n'.join(['p$_{{{0:d}}}$: {1:.3f}'.format(i, p)
                                               for i, p in enumerate(fitobj.params)])
                    chi2_params = u'$\chi^2$: {0:.3f}\n\n'.format(fitobj.chi2_min) + param_string

                    plt.annotate(chi2_params, xy=(0.05, 0.95), xycoords='axes fraction',
                                 horizontalalignment='left', verticalalignment='top',
                                 bbox=dict(alpha=0.75, facecolor='white', edgecolor='none'))

                except Exception as mes:  # pylint: disable=broad-except
                    print("Something wrong with fit: {0}".format(mes))

        if output_file:
            plt.savefig(output_file)
        elif show_plot:
            plt.show()
        else:
            pass