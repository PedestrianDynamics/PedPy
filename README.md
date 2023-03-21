
<div align="center">
    <img src="docs/source/_static/logo_text.svg" height="100px" alt="PedPy Logo">
</div>

-----------------
[![PyPI Latest Release](https://img.shields.io/pypi/v/pedpy.svg)](https://pypi.org/project/pedpy/)
[![Nightly Release](https://img.shields.io/badge/nightly-install-9cf)](https://test.pypi.org/project/PedPy/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pedpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7386931.svg)](https://doi.org/10.5281/zenodo.7386931)
[![License](https://img.shields.io/pypi/l/pedpy.svg)](https://github.com/PedestrianDynamics/pedpy/blob/main/LICENSE)
![ci workflow](https://github.com/PedestrianDynamics/pedestrian-trajectory-analyzer/actions/workflows/ci.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Documentation Status](https://readthedocs.org/projects/pedpy/badge/?version=latest)](http://pedpy.readthedocs.io/?badge=latest)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7046/badge)](https://bestpractices.coreinfrastructure.org/projects/7046)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)

# PedPy: Analysis of pedestrian dynamics based on trajectory files.  

`PedPy` is a python module for pedestrian movement analysis. 
It implements different measurement methods for density, velocity and flow.

This repo is a port from the original `JPSreport` to a Python implementation, and will provide the same functionalities.

## Getting started
### Setup Python
For setting up your Python Environment a Python version >= 3.8 is recommended (our code is tested with 3.8 and 3.10).
To avoid conflicts with other libraries/applications the usage of virtual environments is recommended, see [Python Documentation](https://docs.python.org/3/library/venv.html) for more detail.

### Installing PedPy
To install the latest **stable** version of `PedPy` and its dependencies from PyPI:
```bash
python3 -m pip install pedpy
```

If you want to install the current version in the repository which might be unstable, you can do so via:
```bash
python3 -m pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pedpy
```

### Usage

The general usage of `PedPy` is demonstrated in the [usage notebook](notebooks/usage.ipynb).
The [JPSreport notebook](notebooks/jpsreport.ipynb) shows how to use `PedPy` to compute the same results as in the different JPSreport methods.

You can either download the notebooks and [demo files](demos/) from the GitHub repository or clone the whole repository with:
```bash 
git clone https://github.com/PedestrianDynamics/pedpy.git
```

For using either of the notebook some additional libraries need to be installed, mainly for plotting.
You can install the needed libraries with:

```bash
python3 -m pip install jupyter matplotlib
```

Afterwards you can start a jupyter server with:

```bash
jupyter notebook
```

After navigating to one of the notebooks, you can see how the library can be used for different kinds of analysis.

Some examples how the computed values can be visualized are also shown in the notebooks, e.g., density/velocity profiles, fundamental diagrams, N-T-diagrams, etc.

![voronoi](figs/voronoi_diagrams.png)

![density](figs/density_comparison.png)
