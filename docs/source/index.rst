*************************************
PedPy: Pedestrian Trajectory Analyzer
*************************************

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started <getting_started>
   User Guide <user_guide/index>
   API Reference <api/index>
   Methods <methods/index>
   Developer Guide <developer_guide>
   Cite PedPy <citation>
   Changelog <changelog>
   Disclaimer <disclaimer>


**Useful links**:
`Python Packages <https://pypi.org/project/pedpy>`__ |
`Source Repository <https://github.com/PedestrianDynamics/pedpy>`__ |
`Helmholtz RSD <https://helmholtz.software/software/pedpy>`__ |
`Issues <https://github.com/PedestrianDynamics/pedpy/issues>`__ |
`Discussions <https://github.com/PedestrianDynamics/pedpy/discussions>`__

*PedPy* is an open source, MIT-licensed `Python <https://www.python.org/>`__ library for pedestrian movement analysis.
It provides a high-level interface for extracting fundamental measurements from trajectory data.

Basic measures as **velocity**, **flow**, and **density** can directly be computed and then combined to obtain fundamental diagrams.
Additionally, *PedPy* provides functionalities to compute density and velocity **profiles**.

Install PedPy
=============

*PedPy* can directly be installed from PyPI via:

.. code-block::

    pip install pedpy

Cite PedPy
==========

.. include:: ZENODO.rst

|

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting Started
        :img-top: _static/getting_started.svg
        :class-card: intro-card
        :link: getting_started
        :link-type: doc
        :link-alt: To the getting started guide

        New to *PedPy*? Check out the getting started guide.
        It show cases a first analysis done with *PedPy*.

    .. grid-item-card:: User Guide
        :img-top: _static/user_guide.svg
        :class-card: intro-card
        :link: user_guide
        :link-type: doc
        :link-alt: To the user guide

        The user guide shows more detailed applications and provides an overview of the capabilities of *PedPy*.

    .. grid-item-card:: API Reference
        :img-top: _static/api_reference.svg
        :class-card: intro-card
        :link: api/index
        :link-type: doc
        :link-alt: To the reference guide

        A detailed description of *PedPy*'s API.
        It explains the provided functions and their parameters.

    .. grid-item-card:: Developer Guide
        :img-top: _static/developer_guide.svg
        :class-card: intro-card
        :link: developer_guide
        :link-type: doc
        :link-alt: To the development guide

        Want to help improving *PedPy*?
        The developer guides explains, how you can help us to make *PedPy* better.


Software Ecosystem & Tools Using PedPy
======================================

*PedPy* is cataloged in the `Helmholtz Research Software Directory <https://helmholtz.software/software/pedpy>`__ and integrates directly with pedestrian dynamics simulation, tracking, and archival tools:

- `JuPedSim <https://www.jupedsim.org/>`__ (`GitHub <https://github.com/PedestrianDynamics/jupedsim>`__): Uses *PedPy* for trajectory post-processing, fundamental diagram analysis, and model comparison notebooks.
- `PeTrack <https://jugit.fz-juelich.de/ped-dyn-emp/petrack>`__: Extracts experimental pedestrian trajectories from video recordings for quantitative analysis with *PedPy*.
- `Pedestrian Dynamics Data Archive <https://ped.fz-juelich.de/da/doku.php>`__: Open experimental trajectory datasets in HDF5 and text formats supported natively by *PedPy*.
- Interoperable simulation loaders for `Crowd:it <https://www.accu-rate.de/en/software/crowdit/>`__, `Vadere <https://www.vadere.org/>`__, `PTV Viswalk <https://www.ptvgroup.com/en/products/ptv-viswalk>`__, and `Pathfinder <https://www.thunderheadeng.com/pathfinder/>`__.
