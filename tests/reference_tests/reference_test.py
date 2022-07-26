import pathlib

import numpy as np
import pandas as pd
import pygeos
import pytest

from analyzer.data.geometry import Geometry
from analyzer.io.trajectory_parser import parse_trajectory
from analyzer.methods.density_calculator import (
    _compute_individual_voronoi_polygons,
    _compute_intersecting_polygons,
    compute_classic_density,
    compute_voronoi_density,
)
from analyzer.methods.velocity_calculator import (
    compute_mean_velocity_per_frame,
    compute_voronoi_velocity,
)

tolerance = 1e-2


@pytest.mark.parametrize(
    "geometry, measurement_area, folder",
    [
        (
            pygeos.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, 4 -8, -2.25 -8, "
                "-2.25 -0.53, -0.6 -0.53, -0.6 0.53, -2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            pygeos.from_wkt("POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"),
            pathlib.Path("data/bottleneck"),
        )
    ],
)
def test_classic_density(geometry, measurement_area, folder):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/Classical_Voronoi/rho_v_Classic*")),
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


@pytest.mark.parametrize(
    "geometry, measurement_area, folder, velocity_frame",
    [
        (
            pygeos.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, 4 -8, -2.25 -8, "
                "-2.25 -0.53, -0.6 -0.53, -0.6 0.53, -2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            pygeos.from_wkt("POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"),
            pathlib.Path("data/bottleneck"),
            5,
        )
    ],
)
def test_arithmetic_velocity(geometry, measurement_area, folder, velocity_frame):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/Classical_Voronoi/rho_v_Classic*")),
        sep="\t",
        comment="#",
        names=["frame", "classic density", "speed"],
        index_col=0,
        usecols=["frame", "speed"],
    )

    trajectory = parse_trajectory(folder / "traj.txt")

    result, _ = compute_mean_velocity_per_frame(
        trajectory.data, measurement_area, trajectory.frame_rate, velocity_frame
    )
    result = result.to_frame()

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(result["speed"], reference_result["speed"], atol=tolerance).all()


@pytest.mark.parametrize(
    "geometry_polygon, measurement_area, folder",
    [
        (
            pygeos.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, 4 -8, -2.25 -8, "
                "-2.25 -0.53, -0.6 -0.53, -0.6 0.53, -2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            pygeos.from_wkt("POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"),
            pathlib.Path("data/bottleneck"),
        )
    ],
)
def test_voronoi_density(geometry_polygon, measurement_area, folder):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/Classical_Voronoi/rho_v_Voronoi*")),
        sep="\t",
        comment="#",
        names=["frame", "voronoi density", "speed"],
        index_col=0,
        usecols=["frame", "voronoi density"],
    )

    trajectory = parse_trajectory(folder / "traj.txt")
    geometry = Geometry(geometry_polygon)
    result, _ = compute_voronoi_density(trajectory.data, measurement_area, geometry)

    # in jpsreport not all frames are written to the result (e.g., when not enough peds inside ma),
    # hence only compare these who are in reference frame and check if the rest is zero
    assert np.in1d(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result.iloc[reference_result.index]["voronoi density"],
        reference_result["voronoi density"],
        atol=tolerance,
    ).all()
    assert (result.loc[~result.index.isin(reference_result.index)].values == 0).all()


@pytest.mark.parametrize(
    "geometry_polygon, measurement_area, folder, velocity_frame",
    [
        (
            pygeos.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, 4 -8, -2.25 -8, "
                "-2.25 -0.53, -0.6 -0.53, -0.6 0.53, -2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            pygeos.from_wkt("POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"),
            pathlib.Path("data/bottleneck"),
            5,
        )
    ],
)
def test_voronoi_velocity(geometry_polygon, measurement_area, folder, velocity_frame):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/Classical_Voronoi/rho_v_Voronoi*")),
        sep="\t",
        comment="#",
        names=["frame", "voronoi density", "voronoi speed"],
        index_col=0,
        usecols=["frame", "voronoi speed"],
    )

    trajectory = parse_trajectory(folder / "traj.txt")
    geometry = Geometry(geometry_polygon)

    individual_voronoi = _compute_individual_voronoi_polygons(trajectory.data, geometry)
    intersecting_voronoi = _compute_intersecting_polygons(individual_voronoi, measurement_area)

    result, _ = compute_voronoi_velocity(
        trajectory.data,
        intersecting_voronoi,
        trajectory.frame_rate,
        velocity_frame,
        measurement_area,
    )
    result = result.to_frame()

    # in jpsreport not all frames are written to the result (e.g., when not enough peds inside ma),
    # hence only compare these who are in reference frame and check if the rest is zero
    assert np.in1d(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result.iloc[reference_result.index]["voronoi speed"],
        reference_result["voronoi speed"],
        atol=tolerance,
    ).all()
    assert (result.loc[~result.index.isin(reference_result.index)].values == 0).all()
