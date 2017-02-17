Data handling with the Data Class
=================================

*Note: This module is still a work-in-progress and the usage of these classes and/or functions may change in the future.*

The following contains examples of how to use the various features in the :py:class:`.Data`, whose purpose is to simplify the handling of neutron scattering data and provide basic analysis and visualization functions.

Initialization: loading data
----------------------------
There are a couple of options to initialize the :py:class:`.Data` object. If your data is in SPICE (HFIR, ORNL), ICE or ICP (NCNR) formats then you can use :py:meth:`.load`, otherwise you will need to load the data and pass it to :py:class:`.Data` manually.

Load from file
^^^^^^^^^^^^^^
First, let's assume that you have a data file, ``'scan.dat'`` from a HFIR instrument (SPICE format):

>>> from neutronpy.fileio import load_data
>>> data = load_data('scan.dat')

This builds ``data`` automatically and loads all of the data columns from the file.

If you want to load more than one file at a time, simple add the file names as a list or a tuple of file names, *e.g.*

>>> data = load_data(('scan1.dat', 'scan2.dat'))

By default, :py:meth:`.load_data` attempts to determine the format of the input files automatically, but you can specify the ``filetype`` if desired. Valid filetypes are currently:

* ``'auto'`` - Default: attempt to automatically determine the file type
* ``'SPICE'`` - HFIR files
* ``'ICE'`` - NCNR files
* ``'ICP'`` - NCNR files
* ``'DCS_MSLICE'`` - ASCII files exported from DCS MSLICE in Dave
* ``'MAD'`` - ILL files
* ``'GRASP'`` - ASCII and HDF5 files exported from GRASP

Pass pre-loaded data
^^^^^^^^^^^^^^^^^^^^
If your data is not in a format supported by :py:meth:`.Data.load_data` you will need to load the data yourself and pass it to the :py:class:`.Data` class and build ``Data.Q`` using :py:meth:`.Data.build_Q`. To build ``Data.Q`` you must have defined ``h``, ``k``, ``l``, ``e``, and ``temp``.

>>> data = Data(h=h, k=k, l=l, e=e, temp=temp, detector=detector,
                monitor=monitor, time=time)

However, if you do not need to build ``Data.Q``, and want to use some subset of the predefined columns (``h``, ``k``, ``l``, ``e``, ``temp``, ``monitor``, ``detector``, ``time``, the undefined columns will be fill by np.zeros, with the length of the data passed.

>>> data = Data(h=np.arange(0, 1, 0.1))

.. note::
    In the future, it may be possible to pass arbitrary data to build a bare ``Data`` object with the ``Data.data`` attribute. It is already technically possible to do this by building an empty Data object, and reassigning the .data attribute.

    >>> data = Data()
    >>> data.data = dict(angle1=np.arange(47, 49, 0.25))

``Data`` properties
-------------------
Below I outline some of the most common properties that you will want of the :py:class:`.Data` class.

Intensity and error
^^^^^^^^^^^^^^^^^^^
Intensity, *i.e.* ``detector / monitor * m0`` and square-root error are respectively obtained by

>>> data.intensity
>>> data.error

Monitor normalization
^^^^^^^^^^^^^^^^^^^^^
If you want to normalize to a particular monitor ``m0`` then you will need to define it, *e.g.*

>>> data.m0 = 1e5

If you do not choose a ``m0``, when you call :py:meth:`.Data.intensity` one will be defined for you based on the ``monitor`` already defined in ``data``.

Time normalization
^^^^^^^^^^^^^^^^^^
If you want to normalize to a particular time ``t0`` then you will need to set ``time_norm`` to ``True`` and define ``t0`` in minutes, *e.g.*

>>> data.time_norm = True
>>> data.t0 = 5

If you do not choose a ``t0``, when you call :py:meth:`.Data.intensity` one will be defined for you based on the ``time`` already defined in ``data``.

The ``Q`` vector
^^^^^^^^^^^^^^^^
In this case, ``Q`` is collection of column arrays defined as ``[h, k, l, e, temp]``, with ``data.Q.shape = (N, 5)``. Typically, one would expect that ``temp`` not be included in ``Q``, but for the purposes of rebinning it is included currently. *In the future, rebinning may be expanded to include other arbitrary dimensions, rather than just these five.* If data has been loaded from one of the supported file formats, or :py:meth:`.Data.build_Q` has been used then these variables can also be accessed separately by:

>>> h = data.h
>>> k = data.k
>>> l = data.l
>>> e = data.e
>>> temp = data.temp

``Data`` operations
-------------------
Combining data is as easy as adding multiple ``Data`` objects together, *e.g.*

>>> data1 = load_data('scan1.dat', filetype='SPICE')
>>> data2 = load_data('scan2.dat', filetype='SPICE')
>>> data = data1 + data2

This will combine monitor and detector counts for existing points and concatenate unique points in the two objects to create a new ``data`` object.

Subtracting works in a similar way, but keep in mind that in its current form it doesn't interpolate, so if ``Q`` is different between the two ``data`` variables then you will end up with negative intensities at positions where there isn't an overlapping Q. Proper background subtraction will be implemented in the future.

The ``*``, ``/`` and ``**`` operators only act on the detector variable. This is useful for example if you want to apply the detailed balance factor obtained from :py:meth:`.Data.detailed_balance_factor`

Quick analysis
--------------
Often you will want to know the integrated intensity, peak position, and mean-squared width for some part of your data, without relying on fitting. This is easily accomplished with :py:meth:`.Data.integrate`, :py:meth:`.Data.position`, and :py:meth:`.Data.width`.

It is possible to specify the bounds inside which you want to perform these analyses by forming a boolean expression. For example, below is the definition of the bounds of a 1x1 square around (100) at 4 meV:

>>> bounds = ((np.abs(data.h - 1) <= 0.5) & (np.abs(data.k) <= 0.5) &
              (np.abs(data.e - 4) <= 0.25))
>>> int_inten = data.integrate(bounds=bounds)

Binning data
------------
Often data is on an irregular grid with some arbitrary step-size, but you will want to regularly grid your data in some way. You can do this using :py:meth:`.Data.bin`. First you need to define the bin parameters as a dictionary of lists in the form ``[start, end, bins]``. Let's say that we want to bin our data so that we have a ``hk0-e`` volume with 0.025 r.l.u. step size in ``h`` and ``k`` between -2 and 2 r.l.u., and 0.25 meV in ``e`` between -10 and 10 meV, at 300 K for a relatively stable temperature. We would form the bin parameters as follows:

>>> to_bin = {'h': [-2, 2, 161], 'k': [-2, 2, 161], 'l': [-0.2, 0.2, 1],
              'e': [-10, 10, 81], 'temp': [290, 310, 1]}
>>> binned_data = data.bin(to_bin)

The output is a new :py:class:`.Data` object, so that your original data is still maintained in the original `data` object variable.

Visualizing data
----------------
**Note 1**: :py:meth:`.Data.plot` is still relatively experimental. 1-D data plotting and fitting works as intended in its current form, but higher dimensional plotting is still very much a work in progress.

**Note 2**: For publication quality figures, even for 1-D data, it is not recommended to use :py:meth:`.Data.plot`, since some more advanced plot configuration options from matplotlib are not easily available to the user. Instead, :py:meth:`.Data.plot` is currently intended to be used for quickly plotting data for easy visualization.

Basic plotting
^^^^^^^^^^^^^^
Plotting requires at least two parameters to be defined, ``x`` and ``y`` for a line scan plot. By defining ``z`` and ``w`` (or not) you control what type of plot is generated. ``x``, ``y``, ``z``, and ``w`` are defined by assigning one of the following strings: ``'h'``, ``'k'``, ``'l'``, ``'temp'``, ``'e'``, or ``'intensity'``. For example, for a scatter plot with error bars of a line scan, a contour plot of a slice, and a scatter plot of a volume you can do the following, respectively,

>>> data.plot('h', 'intensity')
>>> data.plot('h', 'k', z='intensity')
>>> data.plot('h', 'k', z='e', w='intensity')

Options
^^^^^^^
There are several options that can currently be used to enhance the plots, including rebinning, fitting and smoothing. More options will be added in the future to make the plotting more extensible.

Binning
"""""""
Binning can be achieved by passing the ``bin`` dictionary, as defined in the manner described above in the binning section. For example,

>>> to_bin = {'h': [0.5, 1.5, 41], 'k': [-0.1, 0.1, 1], 'l': [-0.1, 0.1, 1],
              'e': [3.5, 4.5, 1], 'temp': [290, 310, 1]}
>>> data.plot('h', 'intensity', bin=to_bin)

If ``bin`` is not defined, then the raw data is plotted, meaning that if you have multidimensional data that you are trying to plot as a line scan, all of the data will be projected onto the line you want to plot.

Fitting
"""""""
Fitting to arbitrary functions, only applicable for line scan plots, can be performed by passing the ``fit_options`` dictionary. At a minimum, the initial parameters ``p`` and the ``function`` must be defined. Additionally, if holding a parameter fixed is desired, ``fixp`` must be defined as a ``list`` of the same length as ``p`` where ``1`` indicates fixed and ``0`` indicates released. For example,

>>> from neutronpy.functions import gaussian
>>> data.plot('h', 'intensity', fit_options={'p': [0, 0, 1, 0.9, 0.06],
              'function': gaussian, 'fixp': [1, 1, 0, 0, 0]})

Smoothing
"""""""""
Smoothing using a multidimensional gaussian filter can be enabled by passing the ``smooth_options`` dictionary with at least a non-zero ``sigma`` value. Other appropriate options can be found in the `scipy.ndimage.filters.gaussian_filter <http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`_ definition. For example,

>>> data.plot('h', 'intensity', smooth_options={'sigma': 1.0})

Plot options
""""""""""""
Matplotlib plot options may be passed as a dictionary ``plot_options`` to :py:meth:`.Data.plot` for the appropriate plot type:

* Line scan : `errorbar <http://matplotlib.org/api/pyplot_api.html?highlight=errorbar#matplotlib.pyplot.errorbar>`_
* Slice : `pcolormesh <http://matplotlib.org/api/pyplot_api.html?highlight=pcolormesh#matplotlib.pyplot.pcolormesh>`_
* Volume : `scatter <http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html?highlight=scatter#mpl_toolkits.mplot3d.Axes3D.scatter>`_

Miscellaneous
"""""""""""""
* ``show_plot`` : If False, ``plt.show()`` will not be executed inside the :py:meth:`.Data.plot` method, and will have to be executed separately. Useful if overplotting.
* ``output_file`` : If defined, a file with the plot will be saved, in the format specified by the file extension. File type must be supported by the active `matplotlib backend <http://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`_
* ``show_err`` : If False, will not plot error bars on the scan line plot.
