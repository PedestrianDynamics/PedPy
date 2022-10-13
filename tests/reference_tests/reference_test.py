import pathlib

import numpy as np
import pandas as pd
import pytest
import shapely

from analyzer import TrajectoryUnit
from analyzer.data.geometry import Geometry
from analyzer.io.trajectory_loader import load_trajectory
from analyzer.methods.density_calculator import (
    _compute_individual_voronoi_polygons,
    _compute_intersecting_polygons,
    compute_classic_density,
    compute_passing_density,
    compute_voronoi_density,
)
from analyzer.methods.flow_calculator import compute_flow, compute_n_t
from analyzer.methods.method_utils import compute_frame_range_in_area
from analyzer.methods.profile_calculator import VelocityMethod, compute_profiles
from analyzer.methods.velocity_calculator import (
    compute_individual_velocity,
    compute_mean_velocity_per_frame,
    compute_passing_speed,
    compute_voronoi_velocity,
)

TOLERANCE = 1e-2

ROOT_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "geometry, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            shapely.from_wkt(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            shapely.from_wkt(
                "POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"
            ),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            shapely.from_wkt("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_classic_density(geometry, measurement_area, folder):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/rho_v_Classic*"
            )
        ),
        sep="\t",
        comment="#",
        names=["frame", "classic density", "classic velocity"],
        index_col=0,
        usecols=["frame", "classic density"],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    result = compute_classic_density(trajectory.data, measurement_area)

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["classic density"],
        reference_result["classic density"],
        atol=TOLERANCE,
    ).all()


@pytest.mark.parametrize(
    "geometry, measurement_area, folder, velocity_frame",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            shapely.from_wkt(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            5,
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            shapely.from_wkt(
                "POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"
            ),
            ROOT_DIR / pathlib.Path("data/corridor"),
            5,
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            shapely.from_wkt("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
            5,
        ),
    ],
)
def test_arithmetic_velocity(
    geometry, measurement_area, folder, velocity_frame
):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/rho_v_Classic*"
            )
        ),
        sep="\t",
        comment="#",
        names=["frame", "classic density", "speed"],
        index_col=0,
        usecols=["frame", "speed"],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    result, _ = compute_mean_velocity_per_frame(
        trajectory.data, measurement_area, trajectory.frame_rate, velocity_frame
    )
    result = result.to_frame()

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["speed"], reference_result["speed"], atol=TOLERANCE
    ).all()


@pytest.mark.parametrize(
    "geometry_polygon, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            shapely.from_wkt(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            shapely.from_wkt(
                "POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"
            ),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            shapely.from_wkt("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_voronoi_density(geometry_polygon, measurement_area, folder):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/rho_v_Voronoi*"
            )
        ),
        sep="\t",
        comment="#",
        names=["frame", "voronoi density", "speed"],
        index_col=0,
        usecols=["frame", "voronoi density"],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )
    geometry = Geometry(geometry_polygon)
    result, _ = compute_voronoi_density(
        trajectory.data, measurement_area, geometry
    )

    # in jpsreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.in1d(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)]["voronoi density"],
        reference_result["voronoi density"],
        atol=TOLERANCE,
    ).all()
    assert (
        result.loc[~result.index.isin(reference_result.index)].values == 0
    ).all()


@pytest.mark.parametrize(
    "geometry_polygon, measurement_area, folder, velocity_frame",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            shapely.from_wkt(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, 2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            5,
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            shapely.from_wkt(
                "POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"
            ),
            ROOT_DIR / pathlib.Path("data/corridor"),
            5,
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            shapely.from_wkt("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
            5,
        ),
    ],
)
def test_voronoi_velocity(
    geometry_polygon, measurement_area, folder, velocity_frame
):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/rho_v_Voronoi_Voronoi*"
            )
        ),
        sep="\t",
        comment="#",
        names=["frame", "voronoi density", "voronoi speed"],
        index_col=0,
        usecols=["frame", "voronoi speed"],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )
    geometry = Geometry(geometry_polygon)

    individual_voronoi = _compute_individual_voronoi_polygons(
        trajectory.data, geometry
    )
    intersecting_voronoi = _compute_intersecting_polygons(
        individual_voronoi, measurement_area
    )

    result, _ = compute_voronoi_velocity(
        trajectory.data,
        intersecting_voronoi,
        trajectory.frame_rate,
        velocity_frame,
        measurement_area,
    )
    result = result.to_frame()

    # in jpsreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.in1d(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)]["voronoi speed"],
        reference_result["voronoi speed"],
        atol=TOLERANCE,
    ).all()
    assert (
        result.loc[~result.index.isin(reference_result.index)].values == 0
    ).all()


@pytest.mark.parametrize(
    "line, folder",
    [
        (
            shapely.from_wkt("LINESTRING (-2.25 0.5, 4 0.5)"),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("LINESTRING (0 0, 0 5)"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt("LINESTRING (-3 0, 0 0)"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_nt(line, folder):
    reference_result = pd.read_csv(
        next(folder.glob("results/Fundamental_Diagram/FlowVelocity/Flow_NT*")),
        sep="\t",
        comment="#",
        names=["frame", "Time [s]", "Cumulative pedestrians"],
        index_col=0,
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    result, _ = compute_n_t(trajectory.data, line, trajectory.frame_rate)
    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["Time [s]"], reference_result["Time [s]"], atol=TOLERANCE
    ).all()
    assert (
        result["Cumulative pedestrians"]
        == reference_result["Cumulative pedestrians"]
    ).all()


@pytest.mark.parametrize(
    "line, folder, flow_frame, velocity_frame",
    [
        (
            shapely.from_wkt("LINESTRING (-2.25 0.5, 4 0.5)"),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            150,
            5,
        ),
        (
            shapely.from_wkt("LINESTRING (0 0, 0 5)"),
            ROOT_DIR / pathlib.Path("data/corridor"),
            100,
            5,
        ),
        (
            shapely.from_wkt("LINESTRING (-3 0, 0 0)"),
            ROOT_DIR / pathlib.Path("data/corner"),
            100,
            5,
        ),
    ],
)
def test_flow(line, folder, flow_frame, velocity_frame):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/FlowVelocity/FDFlowVelocity*"
            )
        ),
        sep="\t",
        comment="#",
        names=["Flow rate(1/s)", "Mean velocity(m/s)"],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    individual_speed = compute_individual_velocity(
        trajectory.data, trajectory.frame_rate, velocity_frame
    )
    nt, crossing = compute_n_t(trajectory.data, line, trajectory.frame_rate)
    result = compute_flow(
        nt, crossing, individual_speed, flow_frame, trajectory.frame_rate
    )

    # ignore the first flow value as there is a bug in jpsreport, the first x
    # passing will be not included in the flow, hence it is underestimated
    assert np.isclose(
        result["Mean velocity(m/s)"][1:],
        reference_result["Mean velocity(m/s)"][1:],
        atol=TOLERANCE,
    ).all()
    assert np.isclose(
        result["Flow rate(1/s)"][1:],
        reference_result["Flow rate(1/s)"][1:],
        atol=TOLERANCE,
    ).all()


@pytest.mark.parametrize(
    "measurement_line, width, folder",
    [
        (
            shapely.from_wkt("LINESTRING(-0.6 -0.53, 2.4 -0.53)"),
            1.06,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("LINESTRING (0 0, 0 5)"),
            1.0,
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt("LINESTRING (-3 0, 0 0)"),
            2.0,
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_passing_density(measurement_line, width, folder):
    reference_result = (
        pd.read_csv(
            next(folder.glob("results/Fundamental_Diagram/TinTout/FDTinTout*")),
            sep="\t",
            comment="#",
            names=["ID", "density", "speed"],
            usecols=["ID", "density"],
        )
        .sort_values(by="ID")
        .reset_index(drop=True)
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    frames_in_area, measurement_area = compute_frame_range_in_area(
        trajectory.data, measurement_line, width
    )
    density = compute_classic_density(trajectory.data, measurement_area)
    result = (
        compute_passing_density(density, frames_in_area)
        .sort_values(by="ID")
        .reset_index(drop=True)
    )

    # there are some accuracy differences in jpsreport and pedpy, hence some
    # pedestrians frame range inside the measurement area differ.
    # There pedestrians will be ignored in this test.
    if folder.name == "corridor":
        result = result.drop(result[result.ID == 429].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.ID == 429].index
        )
    if folder.name == "corner":
        result = result.drop(result[result.ID == 25].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.ID == 25].index
        )

    assert reference_result["ID"].equals(result["ID"])
    assert np.isclose(
        result["density"], reference_result["density"], atol=TOLERANCE
    ).all()


@pytest.mark.parametrize(
    "measurement_line, width, folder",
    [
        (
            shapely.from_wkt("LINESTRING(-0.6 -0.53, 2.4 -0.53)"),
            1.06,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("LINESTRING (0 0, 0 5)"),
            1.0,
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt("LINESTRING (-3 0, 0 0)"),
            2,
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_passing_velocity(measurement_line, width, folder):
    reference_result = (
        pd.read_csv(
            next(folder.glob("results/Fundamental_Diagram/TinTout/FDTinTout*")),
            sep="\t",
            comment="#",
            names=["ID", "density", "speed"],
            usecols=["ID", "speed"],
        )
        .sort_values(by="ID")
        .reset_index(drop=True)
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    frames_in_area, _ = compute_frame_range_in_area(
        trajectory.data, measurement_line, width
    )

    result = compute_passing_speed(
        frames_in_area, trajectory.frame_rate, width
    ).reset_index(drop=True)

    # there are some accuracy differences in jpsreport and pedpy, hence some
    # pedestrians frame range inside the measurement area differ.
    # There pedestrians will be ignored in this test.
    if folder.name == "corridor":
        result = result.drop(result[result.ID == 429].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.ID == 429].index
        )
    if folder.name == "corner":
        result = result.drop(result[result.ID == 25].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.ID == 25].index
        )

    assert reference_result["ID"].equals(result["ID"])
    assert np.isclose(
        result["speed"], reference_result["speed"], atol=TOLERANCE
    ).all()


@pytest.mark.parametrize(
    "geometry, grid_size, cut_off_radius, num_edges, blind_points, min_frame, "
    "max_frame, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            0.2,
            0.8,
            12,
            False,
            110,
            120,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            0.2,
            0.8,
            12,
            False,
            110,
            120,
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            0.2,
            0.8,
            12,
            False,
            110,
            120,
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_profiles(
    geometry,
    grid_size,
    cut_off_radius,
    num_edges,
    blind_points,
    min_frame,
    max_frame,
    folder,
):
    density_result_folder = (
        folder / "results/Fundamental_Diagram/Classical_Voronoi/field/density"
    )
    velocity_result_folder = (
        folder / "results/Fundamental_Diagram/Classical_Voronoi/field/velocity"
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )
    trajectory.data = trajectory.data[
        trajectory.data.frame.between(
            min_frame - 5, max_frame + 5, inclusive="both"
        )
    ]
    geo = Geometry(geometry)
    individual_voronoi = _compute_individual_voronoi_polygons(
        trajectory.data, geo, (cut_off_radius, num_edges)
    )
    individual_speed = compute_individual_velocity(
        trajectory.data, trajectory.frame_rate, 5
    )
    combined = pd.merge(
        individual_voronoi, individual_speed, on=["ID", "frame"]
    )

    individual_voronoi_velocity_data = combined[
        combined.frame.between(min_frame, max_frame, inclusive="both")
    ]
    density_profiles, velocity_profiles_arithmetic = compute_profiles(
        individual_voronoi_velocity_data,
        geo.walkable_area,
        grid_size,
        VelocityMethod.ARITHMETIC,
    )
    density_profiles, velocity_profiles_voronoi = compute_profiles(
        individual_voronoi_velocity_data,
        geo.walkable_area,
        grid_size,
        VelocityMethod.VORONOI,
    )
    for frame in range(min_frame, max_frame + 1):
        print(frame)
        reference_density = np.loadtxt(
            next(density_result_folder.glob(f"*{frame}*"))
        )
        assert np.isclose(
            density_profiles[frame - min_frame],
            reference_density,
            atol=TOLERANCE,
        ).all()

        reference_velocity_voronoi = np.loadtxt(
            next(velocity_result_folder.glob(f"*Voronoi*{frame}*"))
        )
        assert np.isclose(
            velocity_profiles_voronoi[frame - min_frame],
            reference_velocity_voronoi,
            atol=TOLERANCE,
        ).all()

        reference_velocity_arithmetic = np.loadtxt(
            next(velocity_result_folder.glob(f"*Arithmetic*{frame}*"))
        )

        # There are artifacts of the polygons going outside the geometry in
        # this test case. They appear to originate from handling the border of
        # polygons differently in shapely and jpsreport (boost::geometry).
        # These fields will be ignored.
        if folder.name == "corner":
            reference_velocity_arithmetic[:25, 15:] = 0
            velocity_profiles_arithmetic[frame - min_frame][:25, 15:] = 0

        assert np.isclose(
            velocity_profiles_arithmetic[frame - min_frame],
            reference_velocity_arithmetic,
            atol=TOLERANCE,
        ).all()
