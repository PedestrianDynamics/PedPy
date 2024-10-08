import pathlib

import numpy as np
import pandas as pd
import pytest
import shapely

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.io.trajectory_loader import TrajectoryUnit, load_trajectory
from pedpy.methods.density_calculator import (
    compute_classic_density,
    compute_passing_density,
    compute_voronoi_density,
)
from pedpy.methods.flow_calculator import compute_flow, compute_n_t
from pedpy.methods.method_utils import (
    Cutoff,
    compute_frame_range_in_area,
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
)
from pedpy.methods.profile_calculator import SpeedMethod, compute_profiles
from pedpy.methods.speed_calculator import (
    SpeedCalculation,
    compute_individual_speed,
    compute_mean_speed_per_frame,
    compute_passing_speed,
    compute_voronoi_speed,
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
        names=[FRAME_COL, DENSITY_COL],
        index_col=0,
        usecols=[FRAME_COL, DENSITY_COL],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    result = compute_classic_density(
        traj_data=trajectory, measurement_area=measurement_area
    )

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result[DENSITY_COL],
        reference_result[DENSITY_COL],
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
def test_arithmetic_speed(
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
        names=[FRAME_COL, DENSITY_COL, SPEED_COL],
        index_col=0,
        usecols=[FRAME_COL, SPEED_COL],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    individual_speed = compute_individual_speed(
        traj_data=trajectory,
        frame_step=velocity_frame,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    result = compute_mean_speed_per_frame(
        traj_data=trajectory,
        measurement_area=measurement_area,
        individual_speed=individual_speed,
    )
    result = result.to_frame()

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result[SPEED_COL], reference_result[SPEED_COL], atol=TOLERANCE
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
        names=[FRAME_COL, DENSITY_COL],
        index_col=0,
        usecols=[FRAME_COL, DENSITY_COL],
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

    # in JPSreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.isin(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)][DENSITY_COL],
        reference_result[DENSITY_COL],
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
        names=[FRAME_COL, DENSITY_COL],
        index_col=0,
        usecols=[FRAME_COL, DENSITY_COL],
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

    # in JPSreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.isin(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)][DENSITY_COL],
        reference_result[DENSITY_COL],
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
        names=[FRAME_COL, DENSITY_COL],
        index_col=0,
        usecols=[FRAME_COL, DENSITY_COL],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )
    walkable_area = WalkableArea(walkable_area_polygon)
    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=True,
        cut_off=Cutoff(radius=0.8, quad_segments=3),
    )
    result, _ = compute_voronoi_density(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
    )

    # in JPSreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.isin(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)][DENSITY_COL],
        reference_result[DENSITY_COL],
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
def test_voronoi_speed(
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
        names=[FRAME_COL, DENSITY_COL, SPEED_COL],
        index_col=0,
        usecols=[FRAME_COL, SPEED_COL],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )
    walkable_area = WalkableArea(walkable_area_polygon)

    individual_speed = compute_individual_speed(
        traj_data=trajectory,
        frame_step=velocity_frame,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        use_blind_points=False,
    )

    intersecting_voronoi = compute_intersecting_polygons(
        individual_voronoi_data=individual_voronoi,
        measurement_area=measurement_area,
    )

    # # as blind points were turned off, there are frames where no Voronoi
    # # polygons could be computed but a speed
    # speed_intersection = pd.merge(
    #     individual_speed,
    #     intersecting_voronoi,
    #     on=[ID_COL, FRAME_COL],
    #     how="right",
    # )

    result = compute_voronoi_speed(
        traj_data=trajectory,
        individual_voronoi_intersection=intersecting_voronoi,
        individual_speed=individual_speed,
        measurement_area=measurement_area,
    )

    # in JPSreport not all frames are written to the result (e.g., when not
    # enough peds inside ma), hence only compare these who are in reference
    # frame and check if the rest is zero
    assert np.isin(reference_result.index.values, result.index.values).all()
    assert np.isclose(
        result[result.index.isin(reference_result.index)][SPEED_COL],
        reference_result[SPEED_COL],
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
        names=[FRAME_COL, TIME_COL, CUMULATED_COL],
        index_col=0,
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    result, _ = compute_n_t(
        traj_data=trajectory,
        measurement_line=line,
    )

    # In JPSreport crossing was counted when a pedestrians touches the
    # measurement line the first time. In PedPy the crossing is counted
    # when the movement crossed the line and does not end on it. Hence,
    # some slight modifications in edge need to be done to adapt the
    # reference results.
    if folder.name == "corridor":
        reference_result.loc[243, CUMULATED_COL] -= 1
        reference_result.loc[3082, CUMULATED_COL] -= 1

    assert (reference_result.index.values == result.index.values).all()
    assert np.isclose(
        result[TIME_COL], reference_result[TIME_COL], atol=TOLERANCE
    ).all()
    assert (result[CUMULATED_COL] == reference_result[CUMULATED_COL]).all()


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
        names=[FLOW_COL, MEAN_SPEED_COL],
    )

    trajectory = load_trajectory(
        trajectory_file=folder / "traj.txt", default_unit=TrajectoryUnit.METER
    )

    individual_speed = compute_individual_speed(
        traj_data=trajectory,
        frame_step=velocity_frame,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    nt, crossing = compute_n_t(
        traj_data=trajectory,
        measurement_line=line,
    )
    result = compute_flow(
        nt=nt,
        crossing_frames=crossing,
        individual_speed=individual_speed,
        delta_frame=flow_frame,
        frame_rate=trajectory.frame_rate,
    )

    # ignore the first flow value as there is a bug in JPSreport, the first x
    # passing will be not included in the flow, hence it is underestimated
    assert np.isclose(
        result[MEAN_SPEED_COL][1:],
        reference_result[MEAN_SPEED_COL][1:],
        atol=TOLERANCE,
    ).all()
    assert np.isclose(
        result[FLOW_COL][1:],
        reference_result[FLOW_COL][1:],
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
            names=[ID_COL, DENSITY_COL],
            usecols=[ID_COL, DENSITY_COL],
        )
        .sort_values(by=ID_COL)
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
        .sort_values(by=ID_COL)
        .reset_index(drop=True)
    )

    # there are some accuracy differences in JPSreport and pedpy, hence some
    # pedestrians frame range inside the measurement area differ.
    # There pedestrians will be ignored in this test.
    if folder.name == "corridor":
        result = result.drop(result[result.id == 429].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.id == 429].index
        )
    if folder.name == "corner":
        result = result.drop(result[result.id == 25].index)
        reference_result = reference_result.drop(
            reference_result[reference_result.id == 25].index
        )

    assert reference_result[ID_COL].equals(result[ID_COL])
    assert np.isclose(
        result[DENSITY_COL], reference_result[DENSITY_COL], atol=TOLERANCE
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
def test_passing_speed(measurement_line, width, folder):
    reference_result = (
        pd.read_csv(
            next(folder.glob("results/Fundamental_Diagram/TinTout/FDTinTout*")),
            sep="\t",
            comment="#",
            names=[ID_COL, DENSITY_COL, SPEED_COL],
            usecols=[ID_COL, SPEED_COL],
        )
        .sort_values(by=ID_COL)
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

    # there are some accuracy differences in JPSreport and pedpy, hence some
    # pedestrians frame range inside the measurement area differ.
    # There pedestrians will be ignored in this test.
    if folder.name == "corridor":
        result = result.drop(result[result[ID_COL] == 429].index)
        reference_result = reference_result.drop(
            reference_result[reference_result[ID_COL] == 429].index
        )
    if folder.name == "corner":
        result = result.drop(result[result[ID_COL] == 25].index)
        reference_result = reference_result.drop(
            reference_result[reference_result[ID_COL] == 25].index
        )

    assert reference_result[ID_COL].equals(result[ID_COL])
    assert np.isclose(
        result[SPEED_COL], reference_result[SPEED_COL], atol=TOLERANCE
    ).all()


@pytest.mark.parametrize(
    "walkable_area_polygon, grid_size, cut_off_radius, quad_segments, min_frame, "
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
            3,
            110,
            120,
            ROOT_DIR / pathlib.Path("data/bottleneck"),
        ),
        (
            shapely.from_wkt("POLYGON ((-10 0, -10 5, 10 5, 10 0, -10 0))"),
            0.2,
            0.8,
            3,
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
            3,
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
    quad_segments,
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
    )

    walkable_area = WalkableArea(walkable_area_polygon)
    individual_voronoi = compute_individual_voronoi_polygons(
        traj_data=trajectory,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=cut_off_radius, quad_segments=quad_segments),
        use_blind_points=False,
    )
    individual_speed = compute_individual_speed(
        traj_data=trajectory,
        frame_step=frame_step,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    combined = pd.merge(
        individual_voronoi, individual_speed, on=[ID_COL, FRAME_COL]
    )
    combined = combined.merge(trajectory.data, on=[ID_COL, FRAME_COL])

    individual_voronoi_speed_data = combined[
        combined.frame.between(min_frame, max_frame, inclusive="both")
    ]
    density_profiles, speed_profiles_arithmetic = compute_profiles(
        data=individual_voronoi_speed_data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        speed_method=SpeedMethod.ARITHMETIC,
    )
    density_profiles, speed_profiles_voronoi = compute_profiles(
        data=individual_voronoi_speed_data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        speed_method=SpeedMethod.VORONOI,
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

        reference_speed_voronoi = np.loadtxt(
            next(velocity_result_folder.glob(f"*Voronoi*{frame}*"))
        )
        assert np.isclose(
            speed_profiles_voronoi[frame - min_frame],
            reference_speed_voronoi,
            atol=TOLERANCE,
        ).all()

        reference_speed_arithmetic = np.loadtxt(
            next(velocity_result_folder.glob(f"*Arithmetic*{frame}*"))
        )
        print(
            np.max(
                np.linalg.norm(
                    speed_profiles_arithmetic[frame - min_frame]
                    - reference_speed_arithmetic
                )
            )
        )

        # There are artifacts of the polygons going outside the geometry in
        # this test case. They appear to originate from handling the border of
        # polygons differently in shapely and JPSreport (boost::geometry).
        # These fields will be ignored.
        if folder.name == "corner":
            reference_speed_arithmetic[:25, 15:] = 0
            speed_profiles_arithmetic[frame - min_frame][:25, 15:] = 0

        assert np.isclose(
            speed_profiles_arithmetic[frame - min_frame],
            reference_speed_arithmetic,
            atol=TOLERANCE,
        ).all()
