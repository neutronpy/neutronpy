# NeutronPy

**NeutronPy** is a python library with commonly used tools for neutron scattering measurements, including 

	* Structure factor calculation
	* Single-ion magnetic form factor calculation
	* Triple Axis Spectrometer resolution function calculation (Translated from ResLib)
	* Least-Squares fitting (KMPFIT, based on the C-implementation of MPFIT, from the Kapteyn package)
	* And More...

## Requirements

The following packages are required to install this library:

* Python >= 2.7
* numpy >= 1.8.1
* scipy >= 0.14.0

## Copyright & Licensing

Copyright (c) 2014, David M. Fobes, Released under terms in LICENSE

KMPFIT and MPFIT are used in part from the Kapteyn package and a custom implementation of the MINPACK-1 Least Squares Fitting Library in C, released under the terms in LICENSE.KAPTEYN and LICENSE.MPFIT, respectively.