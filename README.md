<div align="center">
    <img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/docs/source/_static/logo_text.svg" height="100px" alt="PedPy Logo">
</div>

<div align="center">

[![PyPI Latest Release](https://img.shields.io/pypi/v/pedpy.svg)](https://pypi.org/project/pedpy/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pedpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg)](https://doi.org/10.5281/zenodo.7194992)
[![License](https://img.shields.io/pypi/l/pedpy.svg)](https://github.com/PedestrianDynamics/pedpy/blob/main/LICENSE)
![ci workflow](https://github.com/PedestrianDynamics/PedPy/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/PedestrianDynamics/PedPy/graph/badge.svg?token=X5C9NTKAVK)](https://codecov.io/gh/PedestrianDynamics/PedPy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation Status](https://readthedocs.org/projects/pedpy/badge/?version=latest)](http://pedpy.readthedocs.io/?badge=latest)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7046/badge)](https://bestpractices.coreinfrastructure.org/projects/7046)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![JuRSE Code Pick – July 2025](https://img.shields.io/badge/JuRSE_Code_Pick-Jul_2025-blue)](https://www.fz-juelich.de/en/rse/jurse-community/jurse-code-of-the-month/july-2025)

</div>

# PedPy

**PedPy** is a Python library for quantitative analysis of pedestrian dynamics from trajectory data.

It provides a high-level interface for extracting fundamental measures (density, velocity, flow) and advanced analyses such as Voronoi-based methods, profiles, and pair-distribution functions, which can be combined to derive fundamental diagrams.


## Features

- Compute core pedestrian measures: density, velocity, flow
- Advanced analyses: Voronoi-based density, profiles, pair-distribution functions
- Directly load trajectory data from multiple tools: Crowdit, Viswalk, JuPedSim, Vadere, Pathfinder
- Easy-to-use API for loading, processing, and visualizing data
- Built-in plotting for quick inspection and comparison of results
- Open-source, tested, and aligned with FAIR and OpenSSF best practices

## Getting Started

### Installation

**PedPy** requires Python >= 3.11.  
It is recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html).

Install the latest stable release from PyPI:

```bash
python3 -m pip install pedpy
```

To install the latest development version from the repository:

```bash
python3 -m pip install --force-reinstall git+https://github.com/PedestrianDynamics/PedPy.git
```

> [!IMPORTANT]  
> The latest repository version may be unstable. Use with caution.

### Quickstart

```python
from pedpy import *

# Load trajectory data from file
traj = load_trajectory(
  trajectory_file=pathlib.Path("some_trajectory_data.txt")
)

# Create measurement area
measurement_area = MeasurementArea(
  [(-0.4, 0.5), (0.4, 0.5), (0.4, 1.3), (-0.4, 1.3)]
)

# Compute classic density in the measurement area
classic_density = compute_classic_density(
    traj_data=traj, measurement_area=measurement_area
)

plot_density(density=classic_density, title="Classic density")
```

See the [Getting Started Guide](https://pedpy.readthedocs.io/stable/getting_started.html) for a step-by-step introduction.
A more extensive documentation and demonstration of **PedPy**'s capabilities can be found in the [User Guide](https://pedpy.readthedocs.io/stable/user_guide.html).


### Usage

PedPy is designed to be used in scripts or interactive Jupyter notebooks.

- Explore [getting started](notebooks/getting_started.ipynb), [user guide](https://github.com/PedestrianDynamics/PedPy/blob/main/notebooks/user_guide.ipynb), and [fundamental diagram](https://github.com/PedestrianDynamics/PedPy/blob/main/notebooks/fundamental_diagram.ipynb) notebooks.
- For local usage, clone the repository and install the extra requirements for notebooks and plotting:

    ```bash
    git clone https://github.com/PedestrianDynamics/pedpy.git
    python3 -m pip install jupyter matplotlib
    ```

    Then start a Jupyter server:

    ```bash
    jupyter notebook
    ```

## Example Visualizations


<div align="center">
<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="border: none;"><img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/figs/bottleneck_setup.png" width="400" alt="Bottleneck Setup Example"/></td>
    <td style="border: none;">
      <img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/figs/voronoi_cells.png" width="400" alt="Voronoi-based Density Analysis"/>
      <img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/figs/speed_density_profile.png" width="400" alt="Speed-Density Profile"/>
    </td>
  </tr>
</table>
<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="border: none;" width="50%"><img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/figs/density_comparison.png" width="500" alt="Density Comparison Methods"/></td>
    <td style="border: none;" width="50%"><img src="https://raw.githubusercontent.com/PedestrianDynamics/PedPy/refs/heads/main/figs/time_distance.png" width="500" alt="Time-Distance Analysis"/></td>
  </tr>
</table>
</div>


## Documentation

- [Full Documentation](https://pedpy.readthedocs.io/)
- [Getting Started Guide](https://pedpy.readthedocs.io/stable/getting_started.html)
- [Extensive User Guide](https://pedpy.readthedocs.io/stable/user_guide.html)
- [API Reference](https://pedpy.readthedocs.io/stable/api/index.html)


## Citation

If you use **PedPy** in your work, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7194992.svg)](https://doi.org/10.5281/zenodo.7194992)

For the latest release (v.1.3.2) the BibTeX entry is:
```
@software{schrodter_2025_15337052,
  author       = {Schrödter, Tobias and
                  The PedPy Development Team},
  title        = {PedPy - Pedestrian Trajectory Analyzer},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.3.2},
  doi          = {10.5281/zenodo.15337052},
  url          = {https://doi.org/10.5281/zenodo.15337052},
}
```

If you used a different version, please use Zenodo to get the citation information. 

## Contributing

Contributions are welcome and we are looking forward to any contribution from the community!
Take a look at our [Developer Guide](https://pedpy.readthedocs.io/stable/developer_guide.html) to check out different ways to contribute to **PedPy**.
See the [contributing guidelines](CONTRIBUTING.md) and open an issue or pull request on [GitHub](https://github.com/PedestrianDynamics/PedPy/issues).

## Getting Help

If you find yourself in a position where you need assistance from us, don't hesitate to contact us. 
- GitHub Issues: Report bugs or unexpected behavior
- GitHub Discussions: Ask questions, share ideas, request features

## License

**PedPy** is released under the [MIT License](LICENSE).

