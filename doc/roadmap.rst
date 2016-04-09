NeutronPy Roadmap
=================
This document outlines the future of the NeutronPy package, including planned milestones and features. Detail decreases the further in the future we go. The roadmap will be continually updated.

Milestones
----------
There are currently no milestones on Github.

Next Release
------------
In the next release there are several planned features:

* Magnetic Structure Factor calculation
* Space group symmetry operator calculation
    * For use in nuclear & magnetic structure factor calculations
* Polarized cross section corrections
* Data class upgrades
    * Subtraction of measured background
    * Add Spectrometer Angles to Data class
* Bragg peak finder
* Refactor code for easier maintenance
    * ex. File IO separated into new module
    * ex. Energy class to new module
    * ex. New Sample module/class to allow combination of Material, Lattice, and new SpaceGroup classes

Future Features
---------------
This section is for planned features that may require significant development time.

* TAS Scattering plane finder
* Calculation of polarization tensor (spherical polarimetry)
    * Depends on magnetic structure factor calculation
* 3D visualization of volumes (using OpenGL)
    * Support for slicing and cutting
* Sample Activation Calculator
* UI for TAS resolution calcuation
