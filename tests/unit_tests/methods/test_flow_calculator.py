import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementLine
from pedpy.errors import InputError
from pedpy.methods.flow_calculator import compute_line_flow


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


def test_compute_line_flow(example_data):
    species, speed, voronoi, line = example_data

    flow_on_line = compute_line_flow(
        individual_voronoi_polygons=voronoi,
        measurement_line=line,
        species=species,
        individual_speed=speed,
    )

    n = line.normal_vector()
    assert flow_on_line.shape[0] == 1
    assert flow_on_line[FLOW_SP1_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 0.5 * 3)
    assert flow_on_line[FLOW_SP2_COL].values[0] == pytest.approx((n[0] * -1 + n[1] * -1) * 0.5 * -1 * 3)

    assert flow_on_line[FLOW_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 3)


def test_compute_line_flow_invalid_species(example_data):
    """Test that compute_line_flow raises InputError when species data doesn't match Voronoi polygons."""
    species, speed, voronoi, line = example_data

    invalid_species = species.iloc[0:1]
    error_text = "Species data validation failed. "
    with pytest.raises(InputError, match=error_text):
        compute_line_flow(
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
            species=invalid_species,
            individual_speed=speed,
        )


def test_compute_line_flow_missing_speed_entries(example_data):
    species, speed, voronoi, line = example_data
    # Remove some speed entries
    invalid_speed = speed.copy()
    invalid_speed = invalid_speed.iloc[1:]  # Remove first row

    with pytest.raises(InputError, match="Missing speed data entries"):
        compute_line_flow(
            individual_speed=invalid_speed,
            species=species,
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
        )


def test_compute_line_flow_missing_velocity_columns(example_data):
    species, speed, voronoi, line = example_data
    # Remove velocity columns
    invalid_speed = speed.copy()
    invalid_speed = invalid_speed.drop([V_X_COL, V_Y_COL], axis=1)  # Assuming these are your velocity columns

    with pytest.raises(InputError, match="Required velocity columns missing"):
        compute_line_flow(
            individual_speed=invalid_speed,
            species=species,
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
        )


def test_compute_flow_without_individual_speed_and_classic_fd():
    from tests.utils.utils import make_traj

    from pedpy.data.geometry import MeasurementArea
    from pedpy.methods.flow_calculator import (
        compute_classic_fundamental_diagram,
        compute_flow,
        compute_n_t,
    )
    from pedpy.methods.speed_calculator import (
        SpeedCalculation,
        compute_individual_speed,
    )
    from pedpy.plotting.plotting import plot_flow_at_line

    # Three pedestrians walking along +x at 1 m/s across line x=2.0 inside [0,4]x[0,2]
    ids, frames, xs, ys = [], [], [], []
    for pid, offset in enumerate([1.5, 0.5, -0.5]):
        for f in range(10):
            ids.append(pid)
            frames.append(f)
            xs.append(offset + 0.5 * f)
            ys.append(1.0)
    traj = make_traj(ids=ids, frames=frames, xs=xs, ys=ys, frame_rate=2.0)
    line = MeasurementLine([(2.0, 0.0), (2.0, 2.0)])
    area = MeasurementArea([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])

    nt, crossing = compute_n_t(traj_data=traj, measurement_line=line)
    speed = compute_individual_speed(
        traj_data=traj,
        frame_step=1,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    flow_only = compute_flow(
        nt=nt,
        crossing_frames=crossing,
        delta_frame=3,
        frame_rate=traj.frame_rate,
    )
    flow_with_speed = compute_flow(
        nt=nt,
        crossing_frames=crossing,
        individual_speed=speed,
        delta_frame=3,
        frame_rate=traj.frame_rate,
    )
    assert list(flow_only.columns) == [FLOW_COL]
    assert list(flow_with_speed.columns) == [FLOW_COL, MEAN_SPEED_COL]
    np.testing.assert_allclose(flow_only[FLOW_COL], flow_with_speed[FLOW_COL])

    fd = compute_classic_fundamental_diagram(
        traj_data=traj,
        measurement_area=area,
        measurement_line=line,
        delta_frame=3,
    )
    assert list(fd.columns) == [
        FIRST_FRAME_COL,
        LAST_FRAME_COL,
        DENSITY_COL,
        FLOW_COL,
    ]
    np.testing.assert_allclose(fd[FLOW_COL], flow_only[FLOW_COL])
    assert (fd[DENSITY_COL] > 0.0).all()

    fd_speed = compute_classic_fundamental_diagram(
        traj_data=traj,
        measurement_area=area,
        measurement_line=line,
        delta_frame=3,
        individual_speed=speed,
    )
    assert MEAN_SPEED_COL in fd_speed.columns

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sample_line_flow = pd.DataFrame(
        {
            FRAME_COL: [0, 1],
            FLOW_SP1_COL: [0.5, 0.6],
            FLOW_SP2_COL: [0.0, 0.0],
            FLOW_COL: [0.5, 0.6],
        }
    )
    plot_flow_at_line(flow_at_line=sample_line_flow, axes=ax)
    assert "1/(m" in ax.get_ylabel()
    plt.close(fig)
