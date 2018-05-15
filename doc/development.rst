===========
Development
===========

To contribute constructively to the development of NeutronPy there are certain requirements and recommendations. This document outlines the typical workflow for contributing code to the neutronpy repository on GitHub.

Requirements
------------
Contributors should keep these following requirements in mind while creating fixes or new functionality:

    * Code is to be written in Python 3 and then made backwards compatible with py27.
    * PEP8 conventions should be followed, with some exceptions
    * Documentation should always be included and/or updated
    * Changelog should be updated
    * Unit tests are necessary for new functionality

Write for Python 3
^^^^^^^^^^^^^^^^^^
All contributed code should be written for Python 3 first. While we support Python 2.6 and 2.7, these versions are woefully out of date compared with the most recent commonly-used versions of Python, 3.4 and 3.5, and compatibility with future versions of Python is the number one priority. However, since we do support older versions of Python it is important to restrict oneself to the features contained in the standard library of 2.6 at this time. So far this has caused some minor headaches but there is very little that would truly break compatibility. In the near future we will drop support for Python 2.6 (probably at the time of 1.1.0), but Python 2.7 support will likely continue for many more years, at least as long as it is supported by the Python Software Foundation.

Some frequent issues you may encounter include:

    * File encoding: file should explicitly have the following encoding on the first line: ``# -*- coding: utf-8 -*-``. This is a problem with both 2.6 and 2.7
    * ``*args`` and ``**kwargs`` can only be used like ``function(*args, **kwargs)`` in Python 2, whereas in Python 3 you could for instance do ``function(*args keyword=None)``, explicitly naming a ``kwarg`` after ``*args``.

PEP8 Formatting
^^^^^^^^^^^^^^^
Contributors should attempt to follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ conventions. In certain cases PEP8 conventions are ignored in favor of readability and adherance to neutron scattering naming conventions. Some notable exceptions include:

    * Line lengths can be up to 120 characters instead of the standard 80
    * Variable names can be capitalized or be only one character, e.g. in neutron scattering **Q**, *h*, *k*, *l*, etc. all have well known meanings, so we defer to neutron scattering terminology

It is suggested that you run e.g. ``pep8 --ignore=E501`` on your code to find obvious easy-to-fix PEP8 problems. More detailed analysis with e.g. ``pylint`` can be helpful but is not necessary.

Documentation
^^^^^^^^^^^^^
All classes, methods, attributes, functions, etc. should have complete documentation in the form of docstrings detailing their proper usage. The documentation seen at neutronpy.github.io is automatically generated using :py:mod:`sphinx` and :py:mod:`numpydoc`. This means that documentation should be in the `numpy style <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Docstrings are written in the `reStructuredText (reST) <http://www.sphinx-doc.org/en/stable/rest.html>`_ format, which makes it easy to refer and link to other functions/classes/etc. from inside docstrings.

In order to build documentation you must have

Unit tests
^^^^^^^^^^
In neutronpy, all unit tests are performed using the :py:mod:`unittest` Standard Python Library, and some using :py:mod:`mock` which was only added to the Standard Library in py33, but is available from pypi for earlier versions. See existing tests in the ``tests`` top-level directory for examples.

Development should generally take a test-driven approach, i.e. expected behavior of new functions should be known in advance and tests should be written first. A basic introduction to test-driven development can be found at `tutsplus.com <http://code.tutsplus.com/tutorials/beginning-test-driven-development-in-python--net-30137>`_.

If you plan to make a significant contribution to neutronpy, I highly recommend making at least a `Travis-ci <https://travis-ci.org/>`_ account to run the tests on your commits before you submit a pull request. It will help you make changes before making them public.

Development Workflow
--------------------
The following is an outline of the basic workflow, otherwise known as `"fork and pull" <https://gist.github.com/Chaser324/ce0505fbed06b947d962>`_, which you should follow when you want to contribute to neutronpy:

    1. Fork ``master`` to your account (forked repo) and clone to your machine (local repo)
    2. Create a new branch for your bug-fix/feature/etc with a descriptive name
    3. Make your changes and commit them with a appropriate git commit message
    4. Rebase your branch based on the most up-to-date neutronpy ``master`` branch
    5. Push your branch to your forked repo
    6. Create a pull request from your forked branch to neutronpy ``master``

You will need to have some comfort level with ``git``, for which there are quite a few guides. Github has an `interactive introduction <https://try.github.io/>`_, and a useful `cheat sheet <https://services.github.com/kit/downloads/github-git-cheat-sheet.pdf>`_ with commonly used commands. If you want to use a good visual ``git`` interface, I highly recommend `ungit <https://github.com/FredrikNoren/ungit>`_

Fork and clone
^^^^^^^^^^^^^^
First, `fork <https://github.com/neutronpy/neutronpy#fork-destination-box>`_ the existing neutronpy repo onto your personal github account. Then clone the fork to your local machine using ``git clone git@github.com:username/neutronpy.git``. Now you have a local copy that you can make your changes to.

Create a branch
^^^^^^^^^^^^^^^
With ``master`` checked out, create a new local branch ``git branch [branch-name]`` with a descriptive ``[branch-name]`` that indicates the feature you are adding/changing. A good place to start is with the git-flow naming convention, i.e. ``feature/feature-name``, or ``bugfix/fix-name``. I don't necessarily advocate directly using git-flow, although you certainly may if you are comfortable operating this way, but I like the naming convention because it makes the future pull request easy to parse at a glance.

Commit changes
^^^^^^^^^^^^^^
When you commit your changes to the branch you should write a description title and message for your commit.

Acronyms that should be used for neutronpy include::

    REF: Refactor (incl. PEP8 fixes)
    FIX: Bug fix
    DOC: Documentation update
    ADD: Add new feature
    ENH: Enhance a feature
    UPD: Update existing feature
    REM: Remove a feature
    DEP: Depreciate a feature
    TST: Add or change unit tests
    RLS: Release

Rebase on neutronpy ``master``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you have completed your changes it is possible that the neutronpy ``master`` branch is ahead of the place where you created your local branch. In this case, you will need to update your local ``master`` branch from ``upstream`` and then perform a ``rebase``.

To do this you first add the ``upstream`` repo::

    git remote add upstream https://github.com/neutronpy/neutronpy.git

Next, you fetch the ``upstream`` repo, checkout your local ``master``, and merge. Since you are doing all of your own work on your new branch, this should simply fast forward the master branch::

    git fetch upstream
    git checkout master
    git merge upstream/master

Now you can begin your rebase by checking out your feature branch, and then rebasing::

    git checkout feature-branch
    git rebase master

If any of your changes conflict with the up-to-date master, then you will get some error messages during rebase about **conflicts**. It should give a list of files in conflict and you will need to go there to resolve the conflict. An in-depth guide to resolving rebase conflicts can be found at `gitforteam.com <http://gitforteams.com/resources/rebasing.html>`_.

Create pull request
^^^^^^^^^^^^^^^^^^^
Once you are happy with the changes you have made and would like to submit them to be merged into neutronpy you should push them to your github fork with ``git push origin [branch-name]`` and then create a **Pull Request** (PR) for neutronpy ``master`` using `Github.com <https://github.com>`_. To create a PR you first switch to the branch you want to submit, and then click the "New pull request" button next to it.

Once you have created your PR, your changes will be tested automatically by continuous integration services, Travis-CI (linux and osx) and Appveyor (win). All tests must pass before a PR can be merged into ``master``. If your PR does not pass the tests you must follow steps 3--5 above until it passes. You can keep pushing new commits to the same branch on your own fork and they will automatically be added to the PR.

During this phase there may also be discussion concerning your PR, and you could be asked for example to make changes to conform to style conventions or to established neutronpy coding conventions.

You should also try to keep the number of commits to a minimum. You can start with many commits and then ``git rebase -i master`` to alter the history of the commits since the last one in ``master``, and `squash <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>`_ your changes into a reasonable number of commits. It is also important to rebase to master before This is one of the powers of the Fork and Pull method; you can force push and destroy your own branch history, and neutronpy's history should always remain in tact because until the changes are merged, they are not part of the main repo.

Cleanup
^^^^^^^
Once your PR has been accepted you can now delete the branches you created and update your local repo from upstream.

Development Environment
-----------------------
Often you will want to test your new features while still maintaining the stable version of neutronpy on your machine. In this case you should use a virtual environment, using either `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or `Anaconda <https://www.continuum.io/downloads>`_, which is probably easier. For example, using anaconda you can easily create a conda virtualenv with Python 3.6 called py36 and then activate it to install neutronpy (in linux or osx terminal) by::

    conda config --add channels mmcauliffe
    conda create --yes -q -n py36 python=3.6 numpy scipy pqt5 h5py matplotlib
    source activate py34
    pip install neutronpy

Coding can be done in any plaintext text editor, but if you want more features I recommend `PyCharm CE <https://www.jetbrains.com/pycharm/>`_ as an IDE. It is free, easy to use, and handles some of the PEP8 formatting tasks automatically.

Versions
--------
Version number incrementation should follow `Semantic Versioning 2.0.0 <http://semver.org/>`_. This means that the version numbers should follow the general pattern ``X.Y.Z``, where ``X`` is a major version number, indicating a break in backwards compatibility, ``Y`` is a minor version number indicating the addition of features which DO NOT break backwards compatibility, ``Z`` is a patch version number indicating patches which DO NOT add major features or break backwards compatibility. ``X``, ``Y``, and ``Z`` must all be non-negative integer numbers. The more general pattern that a version must match is ``[N!]N(.N)*[{a|b|rc}N][.postN][.devN]``. See `Python Public version identifiers <https://www.python.org/dev/peps/pep-0440/#public-version-identifiers>`_.
