import numpy as np
import pandas as pd
import pytest
from shapely import LineString, Polygon
from shapely.geometry import Point
from tests.utils.utils import get_trajectory, get_trajectory_data

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import InputError, PedPyTypeError, PedPyValueError
from pedpy.methods.density_calculator import (
    SingleFileMethod,
    _get_num_peds_per_frame,
    compute_classic_density,
    compute_line_density,
    compute_single_file_density,
)


@pytest.mark.parametrize(
    "measurement_area, ped_distance, num_ped_col, num_ped_row",
    [
        (MeasurementArea([(-5, -5), (-5, 5), (5, 5), (5, -5)]), 1.0, 5, 5),
        (
            MeasurementArea([(0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)]),
            0.5,
            5,
            5,
        ),
    ],
)
def test_compute_classic_density(measurement_area, ped_distance, num_ped_col, num_ped_row):
    velocity = 1
    movement_direction = np.array([velocity, 0])
    num_frames = 50

    trajectory_data = get_trajectory_data(
        grid_shape=[num_ped_col, num_ped_row],
        number_frames=num_frames,
        movement_direction=movement_direction,
        start_position=np.array([-10, 2 * ped_distance]),
        ped_distance=ped_distance,
        fps=25,
    )

    computed_density = compute_classic_density(traj_data=trajectory_data, measurement_area=measurement_area)

    num_peds_in_area_per_frame = {frame: 0 for frame in range(num_frames)}

    for index, row in trajectory_data.data.iterrows():
        point = Point([row.x, row.y])
        if point.within(measurement_area.polygon):
            num_peds_in_area_per_frame[row.frame] += 1

    expected_density = pd.DataFrame.from_dict(
        {frame: [num_peds / measurement_area.area] for frame, num_peds in num_peds_in_area_per_frame.items()},
        orient="index",
        columns=[DENSITY_COL],
    )

    assert computed_density[FRAME_COL].min() == 0
    assert computed_density[FRAME_COL].max() == num_frames - 1
    assert expected_density[DENSITY_COL].equals(computed_density[DENSITY_COL])


@pytest.mark.parametrize(
    "num_peds_row, num_peds_col, num_frames",
    (
        [
            (4, 5, 100),
            (1, 1, 200),
        ]
    ),
)
def test_get_num_peds_per_frame(num_peds_row, num_peds_col, num_frames):
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[num_peds_col, num_peds_row],
            number_frames=num_frames,
            start_position=np.array([0, 0]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )
    num_peds = num_peds_col * num_peds_row
    num_peds_per_frame = _get_num_peds_per_frame(traj_data)

    assert (num_peds_per_frame[COUNT_COL] == num_peds).all()


@pytest.fixture
def example_data():
    species = pd.DataFrame({ID_COL: [1, 2, 3, 4], SPECIES_COL: [1, -1, np.nan, 1]})
    speed = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [3, 3, 3, 3],
            SPEED_COL: [0, 0, 0, 0],
            V_X_COL: [1, -1, 5, -5],
            V_Y_COL: [1, -1, 5, -5],
        }
    )
    line = MeasurementLine([(2, 0), (0, 2)])
    matching_poly1 = Polygon([(0, 0), (2, 2), (1, 3), (-1, 1)])
    matching_poly2 = Polygon([(1, -1), (3, 1), (2, 2), (0, 0)])
    non_matching_poly = Polygon()
    voronoi = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [3, 3, 3, 3],
            POLYGON_COL: [
                matching_poly1,
                matching_poly2,
                matching_poly1,
                non_matching_poly,
            ],
            DENSITY_COL: [3, 3, 3, 3],
        }
    )
    return species, speed, voronoi, line


def test_compute_line_density(example_data):
    species, speed, voronoi, line = example_data

    density_on_line = compute_line_density(
        individual_voronoi_polygons=voronoi,
        measurement_line=line,
        species=species,
    )

    assert density_on_line.shape[0] == 1
    assert density_on_line[DENSITY_SP1_COL].values[0] == pytest.approx(3 * 0.5)
    assert density_on_line[DENSITY_SP2_COL].values[0] == pytest.approx(3 * 0.5)

    assert density_on_line[DENSITY_COL].values[0] == pytest.approx(3)


def test_compute_line_density_invalid_species(example_data):
    """Test that compute_line_density raises InputError when species data doesn't match Voronoi polygons."""
    species, speed, voronoi, line = example_data

    invalid_species = species.iloc[0:1]
    error_text = "the species data does not contain all information "
    with pytest.raises(InputError, match=error_text):
        compute_line_density(
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
            species=invalid_species,
        )


def test_compute_single_file_density_open_line_voronoi_and_headway():
    # Positions at x = 0, 2, 5, 9 -> interior Voronoi widths: (5-0)/2=2.5, (9-2)/2=3.5
    traj_df = pd.DataFrame(
        {
            ID_COL: [10, 20, 30, 40],
            FRAME_COL: [0, 0, 0, 0],
            X_COL: [0.0, 2.0, 5.0, 9.0],
            Y_COL: [0.0, 0.0, 0.0, 0.0],
        }
    )
    traj = TrajectoryData(data=traj_df, frame_rate=10.0)

    voronoi_df = compute_single_file_density(
        traj_data=traj,
        axis=X_COL,
        closed=False,
        method=SingleFileMethod.VORONOI,
    )
    assert np.isnan(voronoi_df.iloc[0][DENSITY_COL])
    assert np.isnan(voronoi_df.iloc[3][DENSITY_COL])
    assert np.isclose(voronoi_df.iloc[1]["spacing"], 2.5)
    assert np.isclose(voronoi_df.iloc[1][DENSITY_COL], 1.0 / 2.5)
    assert np.isclose(voronoi_df.iloc[2]["spacing"], 3.5)
    assert np.isclose(voronoi_df.iloc[2][DENSITY_COL], 1.0 / 3.5)

    headway_df = compute_single_file_density(
        traj_data=traj,
        axis=X_COL,
        closed=False,
        method=SingleFileMethod.HEADWAY,
    )
    assert np.allclose(headway_df.iloc[:3]["spacing"], [2.0, 3.0, 4.0])
    assert np.allclose(headway_df.iloc[:3][DENSITY_COL], [0.5, 1.0 / 3.0, 0.25])
    assert np.isnan(headway_df.iloc[3][DENSITY_COL])


def test_compute_single_file_density_closed_loop_and_centerline_interval():
    # Closed square centerline of perimeter 16 m with 4 evenly spaced pedestrians (spacing = 4 m, density = 0.25 1/m)
    ring = LineString([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)])
    traj_df = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [0, 0, 0, 0],
            X_COL: [1.0, 4.0, 3.0, 0.0],
            Y_COL: [0.0, 1.0, 4.0, 3.0],
        }
    )
    traj = TrajectoryData(data=traj_df, frame_rate=10.0)

    frame_df, ind_df = compute_single_file_density(
        traj_data=traj,
        centerline=ring,
        closed=True,
        method=SingleFileMethod.VORONOI,
        measurement_interval=(0.0, 8.0),
    )

    assert np.isclose(ind_df["spacing"].sum(), 16.0)
    assert np.allclose(ind_df["spacing"], 4.0)
    assert np.allclose(ind_df[DENSITY_COL], 0.25)
    assert np.allclose(frame_df[DENSITY_COL], [0.25])


def test_compute_single_file_density_validation_errors():
    traj_df = pd.DataFrame({ID_COL: [1, 2], FRAME_COL: [0, 0], X_COL: [0.0, 1.0], Y_COL: [0.0, 0.0]})
    traj = TrajectoryData(data=traj_df, frame_rate=1.0)

    with pytest.raises(PedPyValueError, match="axis"):
        compute_single_file_density(traj_data=traj, axis="z")

    with pytest.raises(PedPyValueError, match="track_length"):
        compute_single_file_density(traj_data=traj, closed=True)

    with pytest.raises(PedPyTypeError, match="LineString"):
        compute_single_file_density(traj_data=traj, centerline="not_a_line")  # type: ignore[arg-type]

    with pytest.raises(PedPyValueError, match="measurement_interval"):
        compute_single_file_density(traj_data=traj, measurement_interval=(5.0, 2.0))
