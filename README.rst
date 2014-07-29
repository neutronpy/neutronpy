NeutronPy
=========
**NeutronPy** is a python library with commonly used tools for neutron scattering measurements, including

* Basic data operations
	* Binning
	* Normalization
	* Calculated peak integrated intensity, position, and width
	* Loading from known filetypes (more to be added)
	* Plotting (incomplete)
	* Slicing (future)
* Structure factor calculation, including:
	* Mass Normalization
	* Debye-Waller factor
	* Unit cell visualization (future)
* Single-ion magnetic form factor calculation
* Triple Axis Spectrometer resolution function calculation (Translated from ResLib), including:
	* Resolution ellipses
	* Instrument visualization (future)
* Least-Squares fitting (KMPFIT, based on the C-implementation of MPFIT, from the Kapteyn package)
* And More...

Requirements
------------
The following packages are required to install this library:

* ``Python >= 2.7.0 (incl. python 3)``
* ``numpy >= 1.8.0``
* ``scipy >= 0.13.0``
* ``matplotlib >= 1.3.0`` (optional, plotting)
* ``sphinx >= 1.2.0`` (optional, documentation)
* ``numpydoc >= 0.5`` (optional, documentation)

Copyright & Licensing
---------------------

Copyright (c) 2014, David M. Fobes, Released under terms in LICENSE

KMPFIT and MPFIT are used in part from the Kapteyn package and a custom implementation of the MINPACK-1 Least Squares Fitting Library in C, released under the terms in LICENSE.KAPTEYN and LICENSE.MPFIT, respectively.

The source for the Triple Axis Spectrometer resolution calculations was translated in part from the ResLib 3.4c (2009) library released under the terms in LICENSE.RESLIB, originally developed by Andrey Zheludev at Brookhaven National Laboratory, Oak Ridge National Laboratory and ETH Zuerich. email: zhelud@ethz.ch.

Disclaimer
----------
THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE COPYRIGHT HOLDERS, THEIR THIRD PARTY LICENSORS, THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.
