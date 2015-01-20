=========
Changelog
=========

* :release:`v0.1.3 <2014-12-30>`
* :bug:`18` Update :py:meth:`.Material.calc_str_fac` to include better checking of input hkl tuple
* :release:`0.1.2 <2014-09-22>`
* :feature:`11` tools package has been changed to core package and package contents are accessible from root level ``neutronpy.``
* :support:`12` Major overhaul of documentation, including new theme based on ReadTheDocs style
* :release:`v0.1.1 <2014-09-12>`
* :bug:`5 major` Update :py:meth:`.Data.bin` to use binary search algorithm for speed increase
* :feature:`6` Added time to :py:class:`.Data` for normalization purposes
* :bug:`7 major` Added tolerances to :py:meth:`.Data.combine_data` so that small differences in Q will be ignored
* :release:`v0.1 <2014-09-09>`
* :feature:`1` Added error plots to :py:class:`.Data` using :py:meth:`.Data.plot`
* :feature:`2` Added fitting to :py:meth:`.Data.plot` with ``fit_options`` argument
* :support:`3` Added examples for :py:class:`.Fitter` to documentation
* :support:`4` Added examples for :py:class:`.Data` to documentation
