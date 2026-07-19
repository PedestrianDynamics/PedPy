===========
User Guide
===========

This user guide shows how to use *PedPy* to analyze pedestrian movement data.
A typical analysis follows the workflow below, and each step is covered by its own notebook.
The notebooks can be read independently, so you can jump directly to the step you are interested in.
Click on a step to open the corresponding notebook.
Steps drawn with a dashed border are optional:

.. mermaid::

    flowchart LR
        MS("<span class='fc-icon'><i class='fa-solid fa-draw-polygon'></i></span><br/><b>Measurement Setup</b><br/>walkable area,<br/>areas and lines")
        LT("<span class='fc-icon'><i class='fa-solid fa-file-import'></i></span><br/><b>Load Trajectory Data</b><br/>import movement data<br/>from files or DataFrames")
        PP("<span class='fc-icon'><i class='fa-solid fa-broom'></i></span><br/><b>Pre-Processing</b><br/>detect outliers and<br/>correct trajectories")
        AN("<span class='fc-icon'><i class='fa-solid fa-chart-line'></i></span><br/><b>Analysis</b><br/>density, speed, flow,<br/>acceleration, profiles, ...")
        FD("<span class='fc-icon'><i class='fa-solid fa-chart-area'></i></span><br/><b>Classic<br/>Fundamental Diagrams</b>")
        FDL("<span class='fc-icon'><i class='fa-solid fa-ruler-horizontal'></i></span><br/><b>Fundamental Diagrams<br/>at Lines</b>")
        WR("<span class='fc-icon'><i class='fa-solid fa-table'></i></span><br/><b>How to work with Results</b><br/>filter, combine and<br/>save the results")

        MS --> PP
        LT --> PP
        PP --> AN
        AN --> FD
        AN --> FDL

        click MS "../notebooks/measurement_setup.html"
        click LT "../notebooks/load_trajectories.html"
        click PP "../notebooks/preprocessing.html"
        click AN "../notebooks/analysis.html"
        click FD "../notebooks/fundamental_diagram.html"
        click FDL "../notebooks/fundamental_diagram_at_measurement_line.html"
        click WR "../notebooks/working_with_results.html"

        classDef step fill:#ffffff,stroke:#4AA3C4,stroke-width:2px,color:#1a1a1a
        classDef showcase fill:#eaf4fb,stroke:#4AA3C4,stroke-width:2px,color:#1a1a1a

        class MS,LT,PP,AN,WR step
        class FD,FDL showcase

        style PP stroke-dasharray: 6 4

.. list-table::
    :widths: 20 80

    * - :doc:`Measurement Setup <../notebooks/measurement_setup>`
      - Define the walkable area with its obstacles, and the measurement areas and lines used in the analysis.
    * - :doc:`Load Trajectories <../notebooks/load_trajectories>`
      - Import pedestrian movement data from the supported file formats (text, HDF5, Viswalk, Vadere, Pathfinder, Crowd:it, JuPedSim) or from a Pandas DataFrame, and validate it against the walkable area.
    * - :doc:`Preprocessing <../notebooks/preprocessing>`
      - Detect and correct outliers and invalid trajectories before the analysis.
    * - :doc:`Analysis <../notebooks/analysis>`
      - Compute density, speed, flow, acceleration, neighborhood, distance to entrance, profiles, RSET maps, and spatial analysis methods.
    * - :doc:`Working with Results <../notebooks/working_with_results>`
      - Handle the data and results with Pandas: filter the data, access specific columns, combine results, and save them to disk.

Showcases
=========

The following notebooks demonstrate complete end-to-end analyses with *PedPy*:

.. list-table::
    :widths: 20 80

    * - :doc:`Fundamental Diagram <../notebooks/fundamental_diagram>`
      - Compute the fundamental diagram of an experiment with the four different measurement methods offered by *PedPy*.
    * - :doc:`Fundamental Diagrams at Measurement Line <../notebooks/fundamental_diagram_at_measurement_line>`
      - Calculate pedestrian flow characteristics using trajectory data, following the methodology in
        `Continuity equation and fundamental diagram of pedestrians <https://arxiv.org/pdf/2409.11857>`_.
        This approach ensures consistency with the continuity equation when analyzing movement patterns
        along measurement lines.

.. toctree::
   :maxdepth: 2
   :hidden:

   Measurement Setup <../notebooks/measurement_setup>
   Load Trajectories <../notebooks/load_trajectories>
   Preprocessing <../notebooks/preprocessing>
   Analysis <../notebooks/analysis>
   Working with Results <../notebooks/working_with_results>
   Fundamental Diagram <../notebooks/fundamental_diagram>
   Fundamental Diagrams at Measurement Line <../notebooks/fundamental_diagram_at_measurement_line>
