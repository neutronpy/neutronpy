Installing NeutronPy
====================
The following explains various methods for installing neutronpy on your system.

Python Package Index - pip
--------------------------
The recommended method for installing `neutronpy <https://pypi.python.org/pypi/neutronpy>`_ is using `pip <https://pip.pypa.io/en/latest/installing.html>`_.

If you do not already have ``pip``, to install it first download `get-pip.py <https://bootstrap.pypa.io/get-pip.py>`_ and run it with the following command::

    python get-pip.py

With ``pip`` installed, you can install the latest version of neutronpy with the command::

    pip install neutronpy

To install a specific version of neutronpy, append ``=={version}`` to the above command, *e.g.*::

    pip install neutronpy==0.1.1

New releases will be pushed to the package index. If you wish to install the development version, you will need to follow the instructions for installation from source.

Installation from Source
------------------------
To install from source, either download the `source from Github <https://github.com/neutronpy/neutronpy/archive/master.zip>`_ or clone the repository::

    git clone https://github.com/neutronpy/neutronpy.git

From inside of the neutronpy directory install the package using::

    python setup.py install

If you want to install the development version, you can either download the `source from Github <https://github.com/neutronpy/neutronpy/archive/develop.zip>`_, as above, after switch to the ``develop`` branch, or clone the repository and checkout the branch and install::

    git clone https://github.com/neutronpy/neutronpy.git
    git fetch
    git checkout develop
    python setup.py install
