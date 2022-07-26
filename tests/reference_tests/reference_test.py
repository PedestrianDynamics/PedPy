import pathlib

import numpy as np
import pandas as pd
import pygeos
import pytest

from analyzer.io.trajectory_parser import parse_trajectory
from analyzer.methods.density_calculator import compute_classic_density

tolerance = 1e-2


@pytest.mark.parametrize(
    "geometry, measurement_area, folder",
    [
        (
            pygeos.from_wkt(
                "POLYGON((40.000 62.500,40.000 5.300,24.000 5.300,24.000 -5.300,40.000 -5.300,40.000 -80.000,-22.500 -80.000,-22.500 -5.300,-6.000 -5.300,-6.000 5.300,-22.500 5.300,-22.500 62.500,40 62.5))"
            ),
            pygeos.from_wkt("POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"),
            pathlib.Path("data/bottleneck"),
        )
    ],
)
def test_classic_density(geometry, measurement_area, folder):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/Classical_Voronoi/rho*")),
        sep="\t",
        comment="#",
        names=["frame", "classic density", "classic velocity"],
        index_col=0,
        usecols=["frame", "classic density"],
    )

    trajectory = parse_trajectory(folder / "traj.txt")

    result = compute_classic_density(trajectory.data, measurement_area)

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["classic density"], reference_result["classic density"], atol=tolerance
    ).all()
