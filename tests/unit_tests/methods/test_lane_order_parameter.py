"""Unit tests for compute_lane_order_parameter in pedpy.methods.spatial_analysis."""

import warnings

import pandas as pd
import pytest
from tests.utils.utils import make_traj

from pedpy.column_identifier import FRAME_COL, ORDER_PARAMETER_COL
from pedpy.errors import PedPyValueError
from pedpy.methods.spatial_analysis import compute_lane_order_parameter

GAMMA = 0.225  # 3r/2 with r = 0.15 m, the value used by von Kruechten (2019)


def make_species(ids: list[int], species: list[float]) -> pd.DataFrame:
    """Builds a species DataFrame from parallel lists."""
    return pd.DataFrame({"id": ids, "species": species})


def test_lane_order_parameter_perfect_lanes():
    """Four pedestrians at y=0.5 walking right, four at y=2.5 walking left.

    The two groups are further apart than gamma, so no pedestrian has an
    opposite-direction lane mate and every individual value is one.
    """
    traj = make_traj(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        frames=[0] * 8,
        xs=[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        ys=[0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5],
    )
    species = make_species(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        species=[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    )

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert len(result) == 1
    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(1.0)


def test_lane_order_parameter_perfect_mixing():
    """Four pedestrians walking right, four walking left, all at y=1.5.

    There are equal number of in-lane and opposite-lane mates, so every individual value becomes zero.
    """
    traj = make_traj(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        frames=[0] * 8,
        xs=[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        ys=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    )
    species = make_species(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        species=[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    )

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert len(result) == 1
    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(0.0)


def test_lane_order_parameter_single_pedestrian():
    """A single pedestrian yields one.

    Each pedestrian is counted as its own lane mate, so the number of
    same-direction lane mates is at least one and the denominator never
    vanishes. An isolated pedestrian therefore scores one. With only one
    pedestrian there is also only one species, so the single-species
    warning is expected here.
    """
    traj = make_traj(ids=[0], frames=[0], xs=[0.0], ys=[1.5])
    species = make_species(ids=[0], species=[1.0])

    with pytest.warns(UserWarning, match="Only one species"):
        result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(1.0)


def test_lane_order_parameter_is_blind_to_which_direction_dominates():
    """Three pedestrians walking right and one walking left, all at y=1.5.

    Every pedestrian sees the same four lane mates, so the majority
    pedestrians have (3 - 1)^2 / (3 + 1)^2 and the single opposing one has
    (1 - 3)^2 / (1 + 3)^2. Squaring makes these equal, and the mean is 0.25.
    """
    traj = make_traj(
        ids=[0, 1, 2, 3],
        frames=[0] * 4,
        xs=[0.0, 1.0, 2.0, 3.0],
        ys=[1.5] * 4,
    )
    species = make_species(ids=[0, 1, 2, 3], species=[1.0, 1.0, 1.0, -1.0])

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(0.25)


def test_lane_order_parameter_is_one_for_sparse_counterflow():
    """Two pedestrians in counterflow, further apart than gamma, yield one.

    Neither pedestrian is a lane mate of the other, so both score one even
    though no lanes have formed. This is the reason values obtained at
    different densities are not directly comparable.
    """
    traj = make_traj(ids=[0, 1], frames=[0, 0], xs=[0.0, 1.0], ys=[0.5, 2.5])
    species = make_species(ids=[0, 1], species=[1.0, -1.0])

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(1.0)


def test_lane_order_parameter_ignores_longitudinal_distance():
    """Pedestrians far apart along the corridor still count as lane mates.

    The lane criterion constrains only the distance perpendicular to the
    walking direction, so a lane extends over the full length of the
    geometry. Here the two groups are 45 m apart and cannot interact, yet
    the result is the same as if they stood next to each other.

    This follows the definition of the order parameter and is intended
    behaviour, but it means the measure can misreport the state of the
    system: here it registers a fully mixed lane, when in fact the two
    groups are too far apart to interact at all. The lane criterion sees
    only the perpendicular coordinate, so it cannot distinguish a genuine
    lane from pedestrians who merely happen to share a height.
    """
    traj = make_traj(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        frames=[0] * 8,
        xs=[0.0, 1.0, 2.0, 3.0, 45.0, 46.0, 47.0, 48.0],
        ys=[1.5] * 8,
    )
    species = make_species(
        ids=[0, 1, 2, 3, 4, 5, 6, 7],
        species=[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    )

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(0.0)


def test_lane_order_parameter_returns_one_row_per_frame():
    """Two frames in which four pedestrians rearrange from mixed to sorted.

    In frame 0 all four share a height and the two directions are equally
    represented, in frame 1 they have separated into two lanes further
    apart than gamma.
    """
    traj = make_traj(
        ids=[0, 1, 2, 3, 0, 1, 2, 3],
        frames=[0, 0, 0, 0, 1, 1, 1, 1],
        xs=[2.0, 2.5, 3.0, 3.5, 2.0, 2.5, 3.0, 3.5],
        ys=[1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 2.5, 2.5],
    )
    species = make_species(ids=[0, 1, 2, 3], species=[1.0, 1.0, -1.0, -1.0])

    result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert list(result[FRAME_COL]) == [0, 1]
    assert result[ORDER_PARAMETER_COL].to_list() == pytest.approx([0.0, 1.0])


def test_duplicate_species_raises():
    """A pedestrian listed twice in the species frame is rejected.

    The merge would otherwise duplicate that pedestrian's row in every
    frame, silently adding a phantom pedestrian.
    """
    traj = make_traj(ids=[0, 1], frames=[0, 0], xs=[0.0, 1.0], ys=[0.5, 2.5])
    species = make_species(ids=[0, 0, 1], species=[1.0, -1.0, -1.0])

    with pytest.raises(PedPyValueError, match="only have one species"):
        compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)


def test_missing_species_raises():
    """A pedestrian absent from the species frame is rejected.

    Such a pedestrian would get a NaN species, which compares unequal to
    everything and makes them an opponent to all.
    """
    traj = make_traj(ids=[0, 1, 2], frames=[0, 0, 0], xs=[0.0, 1.0, 2.0], ys=[0.5, 2.5, 0.5])
    species = make_species(ids=[0, 1], species=[1.0, -1.0])

    with pytest.raises(PedPyValueError, match="No species assigned"):
        compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)


def test_invalid_species_value_raises():
    """A species other than -1 or 1 is rejected.

    compute_species returns np.sign of the speed, which is zero for a
    pedestrian at rest, so a species of zero is a reachable input.
    """
    traj = make_traj(ids=[0, 1], frames=[0, 0], xs=[0.0, 1.0], ys=[0.5, 2.5])
    species = make_species(ids=[0, 1], species=[1.0, 0.0])

    with pytest.raises(PedPyValueError, match="Only species -1 and 1"):
        compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)


def test_single_species_warns_and_returns_one():
    """With one species N_O vanishes for everyone, so Phi is one by algebra.

    Pedestrians 0 and 1 share a lane, pedestrian 2 is isolated, so the result
    holds both for pedestrians with lane mates and without. Two frames are
    used to check the degeneracy holds across the whole output.
    """
    traj = make_traj(
        ids=[0, 1, 2, 0, 1, 2],
        frames=[0, 0, 0, 1, 1, 1],
        xs=[0.0, 1.0, 2.0, 0.1, 1.1, 2.1],
        ys=[0.5, 0.6, 3.0, 0.5, 0.6, 3.0],
    )
    species = make_species(ids=[0, 1, 2], species=[1.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="Only one species"):
        result = compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)

    assert result[ORDER_PARAMETER_COL].iloc[0] == pytest.approx(1.0)


def test_valid_input_does_not_warn():
    """Two well-formed species must not trigger the single-species warning."""
    traj = make_traj(ids=[0, 1], frames=[0, 0], xs=[0.0, 1.0], ys=[0.5, 2.5])
    species = make_species(ids=[0, 1], species=[1.0, -1.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_lane_order_parameter(traj_data=traj, species=species, gamma=GAMMA)
