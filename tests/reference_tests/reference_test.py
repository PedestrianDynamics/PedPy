import pathlib

import numpy as np
import pandas as pd
import pytest
import shapely

from pedpy import TrajectoryUnit
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.io.trajectory_loader import load_trajectory
from pedpy.methods.density_calculator import (
    compute_classic_density,
    compute_passing_density,
    compute_voronoi_density,
)
from pedpy.methods.flow_calculator import compute_flow, compute_n_t
from pedpy.methods.method_utils import (
    compute_frame_range_in_area,
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
)
from pedpy.methods.profile_calculator import VelocityMethod, compute_profiles
from pedpy.methods.velocity_calculator import (
    compute_individual_velocity,
    compute_mean_velocity_per_frame,
    compute_passing_speed,
    compute_voronoi_velocity,
)

TOLERANCE = 1e-2

ROOT_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize(
    "walkable_area, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_classic_density(walkable_area, measurement_area, folder):
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

    result = compute_classic_density(
        traj_data=trajectory, measurement_area=measurement_area
    )

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["classic density"],
        reference_result["classic density"],
        atol=TOLERANCE,
    ).all()


@pytest.mark.parametrize(
    "walkable_area, measurement_area, folder, velocity_frame",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            5,
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
            5,
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
            5,
        ),
    ],
)
def test_arithmetic_velocity(
    walkable_area, measurement_area, folder, velocity_frame
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

    individual_velocity = compute_individual_velocity(
        traj_data=trajectory,
        frame_step=velocity_frame,
    )
    result = compute_mean_velocity_per_frame(
        traj_data=trajectory,
        measurement_area=measurement_area,
        individual_velocity=individual_velocity,
    )
    result = result.to_frame()

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result["speed"], reference_result["speed"], atol=TOLERANCE
    ).all()


@pytest.mark.parametrize(
    "walkable_area_polygon, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53,"
                " 2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_voronoi_density(walkable_area_polygon, measurement_area, folder):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/"
                "rho_v_Voronoi_Voronoi_traj.txt_id_1.dat"
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
    walkable_area = WalkableArea(walkable_area_polygon)

    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=False,
    )
    result, _ = compute_voronoi_density(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
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
    "walkable_area_polygon, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53,"
                " 2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_voronoi_density_blind_points(
    walkable_area_polygon, measurement_area, folder
):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/"
                "rho_v_Voronoi_Voronoi_traj.txt_id_1_blind_points.dat"
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
    walkable_area = WalkableArea(walkable_area_polygon)

    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=True,
    )
    result, _ = compute_voronoi_density(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
    )
    # as there is a bug in the blind point computation in JPSreport in the
    # bottleneck test case ignore the last 20 frames.
    if folder.name == "bottleneck":
        result = result[result.index < 930]
        reference_result = reference_result[reference_result.index < 930]

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
    "walkable_area_polygon, measurement_area, folder",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_voronoi_density_blind_points_cutoff(
    walkable_area_polygon, measurement_area, folder
):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/"
                "rho_v_Voronoi_Voronoi_traj.txt_id_1_blind_points_cut_off.dat"
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
    walkable_area = WalkableArea(walkable_area_polygon)
    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=True,
        cut_off=(0.8, 12),
    )
    result, _ = compute_voronoi_density(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
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
    "walkable_area_polygon, measurement_area, folder, velocity_frame",
    [
        (
            shapely.from_wkt(
                "POLYGON ((4 6.25, 4 0.53, 2.4 0.53, 2.4 -0.53, 4 -0.53, "
                "4 -8.5, -2.25 -8.5, -2.25 -0.53, -0.6 -0.53, -0.6 0.53, "
                "-2.25 0.53, -2.25 6.25, 4 6.25))"
            ),
            MeasurementArea(
                "POLYGON((2.4 0.53, 2.4 -0.53, -0.6 -0.53, -0.6 0.53, "
                "2.4 0.53))"
            ),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            5,
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            MeasurementArea("POLYGON ((-1.5 0, -1.5 5, 1.5 5, 1.5 0, -1.5 0))"),
            ROOT_DIR / pathlib.Path("data/corridor"),
            5,
        ),
        (
            shapely.from_wkt(
                "POLYGON ((0 0, 0 5, -3 5, -3 -3, 5 -3, 5 0, 0 0))"
            ),
            MeasurementArea("POLYGON ((0 0, -3 0, -3 2, 0 2, 0 0))"),
            ROOT_DIR / pathlib.Path("data/corner"),
            5,
        ),
    ],
)
def test_voronoi_velocity(
    walkable_area_polygon, measurement_area, folder, velocity_frame
):
    reference_result = pd.read_csv(
        next(
            folder.glob(
                "results/Fundamental_Diagram/Classical_Voronoi/"
                "rho_v_Voronoi_Voronoi_traj.txt_id_1.dat"
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
    walkable_area = WalkableArea(walkable_area_polygon)

    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=False,
    )
    intersecting_voronoi = compute_intersecting_polygons(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
    )
    individual_velocity = compute_individual_velocity(
        traj_data=trajectory,
        frame_step=velocity_frame,
    )

    result = compute_voronoi_velocity(
        traj_data=trajectory,
        individual_voronoi_intersection=intersecting_voronoi,
        individual_velocity=individual_velocity,
        measurement_area=measurement_area,
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
            MeasurementLine("LINESTRING (-2.25 0.5, 4 0.5)"),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            MeasurementLine("LINESTRING (0 0, 0 5)"),
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            MeasurementLine("LINESTRING (-3 0, 0 0)"),
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

    result, _ = compute_n_t(
        traj_data=trajectory,
        measurement_line=line,
    )
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
            MeasurementLine("LINESTRING (-2.25 0.5, 4 0.5)"),
            ROOT_DIR / pathlib.Path("data/bottleneck"),
            150,
            5,
        ),
        (
            MeasurementLine("LINESTRING (0 0, 0 5)"),
            ROOT_DIR / pathlib.Path("data/corridor"),
            100,
            5,
        ),
        (
            MeasurementLine("LINESTRING (-3 0, 0 0)"),
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
        traj_data=trajectory,
        frame_step=velocity_frame,
    )
    nt, crossing = compute_n_t(
        traj_data=trajectory,
        measurement_line=line,
    )
    result = compute_flow(
        nt=nt,
        crossing_frames=crossing,
        individual_speed=individual_speed,
        delta_t=flow_frame,
        frame_rate=trajectory.frame_rate,
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
            MeasurementLine("LINESTRING(-0.6 -0.53, 2.4 -0.53)"),
            1.06,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            MeasurementLine("LINESTRING (0 0, 0 5)"),
            1.0,
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            MeasurementLine(shapely.from_wkt("LINESTRING (-3 0, 0 0)")),
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
        traj_data=trajectory,
        measurement_line=measurement_line,
        width=width,
    )
    density = compute_classic_density(
        traj_data=trajectory, measurement_area=measurement_area
    )
    result = (
        compute_passing_density(
            density_per_frame=density, frames=frames_in_area
        )
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
            MeasurementLine("LINESTRING(-0.6 -0.53, 2.4 -0.53)"),
            1.06,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            MeasurementLine("LINESTRING (0 0, 0 5)"),
            1.0,
            ROOT_DIR / pathlib.Path("data/corridor"),
        ),
        (
            MeasurementLine("LINESTRING (-3 0, 0 0)"),
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
        traj_data=trajectory,
        measurement_line=measurement_line,
        width=width,
    )

    result = compute_passing_speed(
        frames_in_area=frames_in_area,
        frame_rate=trajectory.frame_rate,
        distance=width,
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
    "walkable_area_polygon, grid_size, cut_off_radius, num_edges, min_frame, "
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
            110,
            120,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            0.2,
            0.8,
            12,
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
            110,
            120,
            ROOT_DIR / pathlib.Path("data/corner"),
        ),
    ],
)
def test_profiles(
    walkable_area_polygon,
    grid_size,
    cut_off_radius,
    num_edges,
    min_frame,
    max_frame,
    folder,
):
    frame_step = 5

    density_result_folder = (
        folder / "results/Fundamental_Diagram/Classical_Voronoi/field/density"
    )
    velocity_result_folder = (
        folder / "results/Fundamental_Diagram/Classical_Voronoi/field/velocity"
    )

    trajectory_original = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    trajectory = TrajectoryData(
        data=trajectory_original.data[
            trajectory_original.data.frame.between(
                min_frame - frame_step, max_frame + frame_step, inclusive="both"
            )
        ],
        frame_rate=trajectory_original.frame_rate,
        file=trajectory_original.file,
    )

    walkable_area = WalkableArea(walkable_area_polygon)
    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        cut_off=(cut_off_radius, num_edges),
        use_blind_points=False,
    )
    individual_speed = compute_individual_velocity(
        traj_data=trajectory,
        frame_step=frame_step,
    )
    combined = pd.merge(
        individual_voronoi, individual_speed, on=["ID", "frame"]
    )

    individual_voronoi_velocity_data = combined[
        combined.frame.between(min_frame, max_frame, inclusive="both")
    ]
    density_profiles, velocity_profiles_arithmetic = compute_profiles(
        individual_voronoi_velocity_data=individual_voronoi_velocity_data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        velocity_method=VelocityMethod.ARITHMETIC,
    )
    density_profiles, velocity_profiles_voronoi = compute_profiles(
        individual_voronoi_velocity_data=individual_voronoi_velocity_data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        velocity_method=VelocityMethod.VORONOI,
    )
    for frame in range(min_frame, max_frame + 1):
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
