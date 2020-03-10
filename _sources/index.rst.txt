.. brahe documentation master file, created by
   sphinx-quickstart on Wed Apr  3 22:08:01 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Brahe Documentation
===================

.. toctree::
    :maxdepth: 1
    :hidden:
        
    modules.rst
    examples.rst

Welcome to the documentation of brahe, a Python satellite astrodynamics, 
guidance, navigation, and control library.

Current high-fidelity satellite GNC modeling software generally runs into the following pitfalls:

1. It is commercially licensed and closed-source code, making it difficult, if not impossible, to be used by hobbyists and academic researchers.
2. Large open-source projects with a steep learning curve making correct use for small projects difficult
3. Out-dated API leading making it hard to incorporate into modern projects.

These challenges make it an unfortunately common occurance that guidance, 
navigation, and control engineers will frequently reimplement common 
astrodynamics libraries for each new project or analysis.

With these deficienties in mind, brahe aims to provide a open-source, MIT-licensed,
high fidelity astrodynamics toolbox to help make it easy to perform high-quality 
simulation and analysis of satellite attitude and orbit dynamics.


This is the sister repository of `SatelliteDynamics.jl <https://github.com/sisl/SatelliteDynamics.jl>`_,
a Julia language implementation of the same functionality. In general, the 
functionality supported by each repository is interchangeable. The main differences
between the repositories come down to the underlying repositories. The Julia 
implementation tends to be more performant, while the Python implementation can
take advantage of the larger package ecosystem.

Getting Started
---------------

To get started with the package simply install it from pypi using pip:

.. code-block:: bash

    pip3 install brahe

Next checkout the various submodules and functions which make up brahe here: :ref:`Modules`

.. Finally, check out examples of brahe in action here: :ref:`Examples`
