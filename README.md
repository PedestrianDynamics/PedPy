
<div align="center">
    <img src="docs/source/_static/logo_text.svg" height="100px" alt="PedPy Logo">
</div>

-----------------
[![PyPI Latest Release](https://img.shields.io/pypi/v/pedpy.svg)](https://pypi.org/project/pedpy/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pedpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg)](https://doi.org/10.5281/zenodo.7194992)
[![License](https://img.shields.io/pypi/l/pedpy.svg)](https://github.com/PedestrianDynamics/pedpy/blob/main/LICENSE)
![ci workflow](https://github.com/PedestrianDynamics/pedestrian-trajectory-analyzer/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/PedestrianDynamics/PedPy/graph/badge.svg?token=X5C9NTKAVK)](https://codecov.io/gh/PedestrianDynamics/PedPy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/pedpy/badge/?version=latest)](http://pedpy.readthedocs.io/?badge=latest)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7046/badge)](https://bestpractices.coreinfrastructure.org/projects/7046)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![JuRSE Code Pick – July 2025](https://img.shields.io/badge/JuRSE_Code_Pick-Jul_2025-blue)](https://www.fz-juelich.de/en/rse/jurse-community/jurse-code-of-the-month/july-2025)

# PedPy: Analysis of pedestrian dynamics based on trajectory files.  

*PedPy* is a python module for pedestrian movement analysis. 
It implements different measurement methods for density, velocity and flow.

If you use *PedPy* in your work, please cite it using the following information from zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg)](https://doi.org/10.5281/zenodo.7194992)


## Getting started

### Setup Python

For setting up your Python Environment a Python version >= 3.11 is recommended (our code is tested with 3.11, 3.12, and 3.13).
To avoid conflicts with other libraries/applications the usage of virtual environments is recommended, see [Python Documentation](https://docs.python.org/3/library/venv.html) for more detail.

### Installing PedPy

To install the latest **stable** version of *PedPy* and its dependencies from PyPI:
```bash
python3 -m pip install pedpy
```

You can also install the latest version of *PedPy* directly from the repository, by following these steps:

1. Uninstall an installed version of *PedPy*:
```bash
python3 -m pip uninstall pedpy
```

2. Install latest version of *PedPy* from repository:
```
python3 -m pip install git+https://github.com/PedestrianDynamics/PedPy.git
```

### Usage

For first time users, have a look at the [getting started notebook](notebooks/getting_started.ipynb), as it shows the first steps to start an analysis with *PedPy*.
A more detailed overview of *PedPy* is demonstrated in the [user guide notebook](notebooks/user_guide.ipynb).
The [fundamental diagram notebook](notebooks/fundamental_diagram.ipynb) shows how to use *PedPy* for computing the fundamental diagram of a series of experiments.

#### Interactive online session

If you want to try out *PedPy* for the first time, you can find an interactive online environments for both notebooks here:

- Getting started: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PedestrianDynamics/PedPy/main?labpath=notebooks%2Fgetting_started.ipynb)
- User guide: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PedestrianDynamics/PedPy/main?labpath=notebooks%2Fuser_guide.ipynb)
- Fundamental diagram: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PedestrianDynamics/PedPy/main?labpath=notebooks%2Ffundamental_diagram.ipynb)

**Note:** 
The execution might be slower compared to a local usage, as only limited resources are available.
It is possible to also upload different trajectory files and run the analysis completely online, but this might not be advisable for long computations.

#### Local usage of the notebooks

For local usage of the notebooks, you can either download the notebooks and [demo files](notebooks/demo-data) from the GitHub repository or clone the whole repository with:
```bash 
git clone https://github.com/PedestrianDynamics/pedpy.git
```

For using either of the notebook some additional libraries need to be installed, mainly for plotting.
You can install the needed libraries with:

```bash
python3 -m pip install jupyter matplotlib
```

Afterward, you can start a jupyter server with:

```bash
jupyter notebook
```

After navigating to one of the notebooks, you can see how the library can be used for different kinds of analysis.

Some examples how the computed values can be visualized are also shown in the notebooks, e.g., density/velocity profiles, fundamental diagrams, N-T-diagrams, etc.

![voronoi](figs/voronoi_diagrams.png)

![density](figs/density_comparison.png)
