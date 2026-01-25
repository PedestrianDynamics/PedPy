**********
User Guide
**********

This user guide showcases the capabilities of *PedPy* through focused, topic-specific notebooks.
Each notebook can be followed independently or as part of a complete workflow.


Workflow Overview
=================

.. raw:: html

    <object data="../_static/user_guide_flow.svg" type="image/svg+xml" style="width: 100%; max-width: 980px;">
      Your browser does not support embedded SVG diagrams. Read the workflow description below.
    </object>

TODO: some text describing what is shown in the different parts

.. `Measurement Setup <../notebooks/measurement_setup.html>`_ introduces the geometric setup, measurement definitions, and how to load trajectory data from various sources.
.. The optional `Pre-processing <../notebooks/preprocessing.html>`_ notebook covers validation, cleaning, and filtering of the data.
.. `Analysis <../notebooks/analysis.html>`_ demonstrates how to compute core pedestrian metrics from processed data.
.. For result-focused workflows, `Fundamental Diagram (FD) <../notebooks/fundamental_diagram.html>`_ explains area-based
.. fundamental diagrams, and `FD at measurement line <../notebooks/fundamental_diagram_at_measurement_line.html>`_ focuses
.. on line-based evaluations. `Working with Results <../notebooks/working_with_results.html>`_ summarizes how to
.. access, combine, and save results.


.. toctree::
   :maxdepth: 2
   :hidden:

   ../notebooks/measurement_setup
   ../notebooks/preprocessing
   ../notebooks/analysis
   ../notebooks/working_with_results
   ../notebooks/fundamental_diagram
   ../notebooks/fundamental_diagram_at_measurement_line
