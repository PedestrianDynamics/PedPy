import pandas as pd
import pytest
import shapely

from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.profile_calculator import RsetMethod, compute_rset_map


def _make_traj(
    ids: list[int],
    frames: list[int],
    xs: list[float],
    ys: list[float],
    frame_rate: float = 1.0,
) -> TrajectoryData:
    df = pd.DataFrame({"id": ids, "frame": frames, "x": xs, "y": ys})
    return TrajectoryData(data=df, frame_rate=frame_rate)


def test_rset_map_single_pedestrian_max():
    """One pedestrian at (0.5, 0.5) at frames 0..4 with fps=1.

    Grid covers [0,2]x[0,2] with grid_size=1 -> 2x2 grid.
    The pedestrian is in the bottom-left cell (x in [0,1], y in [0,1]).
    Max time = 4/1 = 4.0 s.
    """
    n = 5
    traj = _make_traj(
        ids=[0] * n,
        frames=list(range(n)),
        xs=[0.5] * n,
        ys=[0.5] * n,
        frame_rate=1.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MAX,
    )

    assert rset.shape == (2, 2)
    # Bottom-left cell should have max time = 4.0
    assert rset[1, 0] == pytest.approx(4.0)
    # Other cells should be 0 (no observations)
    assert rset[0, 0] == pytest.approx(0.0)
    assert rset[0, 1] == pytest.approx(0.0)
    assert rset[1, 1] == pytest.approx(0.0)


def test_rset_map_min_method():
    """Min method should return the earliest time in each cell."""
    n = 5
    traj = _make_traj(
        ids=[0] * n,
        frames=list(range(n)),
        xs=[0.5] * n,
        ys=[0.5] * n,
        frame_rate=1.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MIN,
    )

    assert rset[1, 0] == pytest.approx(0.0)


def test_rset_map_mean_method():
    """Mean method should return the average time in each cell."""
    n = 5
    traj = _make_traj(
        ids=[0] * n,
        frames=list(range(n)),
        xs=[0.5] * n,
        ys=[0.5] * n,
        frame_rate=1.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MEAN,
    )

    # Mean of [0, 1, 2, 3, 4] = 2.0
    assert rset[1, 0] == pytest.approx(2.0)


def test_rset_map_frame_rate_scaling():
    """Frame rate should scale the time values correctly."""
    n = 10
    traj = _make_traj(
        ids=[0] * n,
        frames=list(range(n)),
        xs=[0.5] * n,
        ys=[0.5] * n,
        frame_rate=10.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MAX,
    )

    # Max frame = 9, fps = 10 -> max time = 0.9 s
    assert rset[1, 0] == pytest.approx(0.9)


def test_rset_map_multiple_pedestrians():
    """Two pedestrians in the same cell; max should pick the latest."""
    traj = _make_traj(
        ids=[0, 0, 0, 1, 1, 1],
        frames=[0, 1, 2, 3, 4, 5],
        xs=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ys=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        frame_rate=1.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MAX,
    )

    assert rset[1, 0] == pytest.approx(5.0)


def test_rset_map_invalid_input():
    """Passing non-TrajectoryData should raise PedPyTypeError."""
    from pedpy.errors import PedPyTypeError

    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    with pytest.raises(PedPyTypeError):
        compute_rset_map(
            traj_data="not_trajectory_data",
            walkable_area=walkable_area,
            grid_size=1.0,
        )


def test_rset_map_two_cells():
    """Pedestrian moves between two cells."""
    traj = _make_traj(
        ids=[0, 0, 0, 0],
        frames=[0, 1, 2, 3],
        xs=[0.5, 0.5, 1.5, 1.5],
        ys=[0.5, 0.5, 0.5, 0.5],
        frame_rate=1.0,
    )
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    rset = compute_rset_map(
        traj_data=traj,
        walkable_area=walkable_area,
        grid_size=1.0,
        method=RsetMethod.MAX,
    )

    # Bottom-left cell: max time = 1.0 s
    assert rset[1, 0] == pytest.approx(1.0)
    # Bottom-right cell: max time = 3.0 s
    assert rset[1, 1] == pytest.approx(3.0)
