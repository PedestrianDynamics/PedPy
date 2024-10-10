import numpy as np
import pandas as pd
import pytest
import shapely

from pedpy.data.geometry import MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import compute_crossing_frames


def create_single_straight_trajectory_crossing_zero_line(
    *, start_x=-10, frames_before, frames_on_line=0, end_x=10, frames_after
):
    x = np.concatenate(
        [
            np.linspace(
                start_x, -0.1 if frames_on_line == 0 else 0, frames_before
            ),
            np.linspace(0, 0, frames_on_line - 1 if frames_on_line > 0 else 0),
            np.linspace(0.1 if frames_on_line == 0 else 0, end_x, frames_after),
        ]
    )
    y = np.zeros_like(x)

    frames = np.arange(0, len(x))
    id = np.ones_like(x)

    df_traj_simple = pd.DataFrame({"id": id, "frame": frames, "x": x, "y": y})
    return TrajectoryData(data=df_traj_simple, frame_rate=1)


@pytest.mark.parametrize(
    "frames_before, frames_after",
    [
        (10, 5),
        (1, 10),
    ],
)
def test_compute_crossing_frame_movement_across_line(
    frames_before, frames_after
):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=frames_before, frames_after=frames_after
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])
    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 1
    assert crossing_frames.iloc[0].id == 1
    assert crossing_frames.iloc[0].frame == frames_before


@pytest.mark.parametrize(
    "frames_before, frames_on_line",
    [
        (3, 1),
        (10, 5),
        (7, 10),
        (0, 1),
        (0, 10),
    ],
)
def test_compute_crossing_frame_movement_stops_on_line(
    frames_before, frames_on_line
):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=frames_before,
        frames_after=5,
        frames_on_line=frames_on_line,
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])

    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 1
    assert crossing_frames.iloc[0].id == 1
    assert crossing_frames.iloc[0].frame == frames_before + frames_on_line


@pytest.mark.parametrize(
    "frames_on_line",
    [
        1,
        5,
        10,
    ],
)
def test_compute_crossing_frame_trajectory_ends_on_line(frames_on_line):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=10,
        frames_after=0,
        frames_on_line=frames_on_line,
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])
    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 0
