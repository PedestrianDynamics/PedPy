"""Module containing functions to compute flows."""
from typing import Tuple

import pandas as pd
import shapely
import numpy

from pedpy.column_identifier import (
    CUMULATED_COL,
    FLOW_COL,
    FRAME_COL,
    ID_COL,
    MEAN_SPEED_COL,
    SPEED_COL,
    TIME_COL,
    V_X_COL,
    V_Y_COL,
    POLYGON_COL,
    DENSITY_COL,
    SPECIES_COL,
    DENSITY_SP1_COL,
    DENSITY_SP2_COL,
    V_SP1_COL,
    V_SP2_COL,
    VELOCITY_COL,
    FLOW_SP1_COL,
    FLOW_SP2_COL,
)
from pedpy.data.geometry import MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import compute_crossing_frames


def compute_n_t(
        *,
        traj_data: TrajectoryData,
        measurement_line: MeasurementLine,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the frame-wise cumulative number of pedestrians passing the line.

    Records the frames, when a pedestrian crossed the given measurement line.
    A frame counts as crossed when the movement is across the line, but does
    not end on it. Then the next frame when the movement starts on the line
    is counted as crossing frame.

    .. warning::

        For each pedestrian only the first passing of the line is considered!

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (MeasurementLine): line for which n-t is computed

    Returns:
        DataFrame containing the columns 'frame', 'cumulative_pedestrians',
        and 'time' since frame 0, and DataFrame containing the columns 'ID',
        and 'frame' which gives the frame the pedestrian crossed the
        measurement line.

    """
    crossing_frames = compute_crossing_frames(
        traj_data=traj_data, measurement_line=measurement_line
    )
    crossing_frames = (
        crossing_frames.groupby(by=ID_COL)[FRAME_COL]
        .min()
        .sort_values()
        .reset_index()
    )

    n_t = (
        crossing_frames.groupby(by=FRAME_COL)[FRAME_COL]
        .size()
        .cumsum()
        .rename(CUMULATED_COL)
    )

    # add missing values, to get values for each frame. First fill everything
    # with the previous valid value (fillna('ffill')). When this is done only
    # the frame at the beginning where no one has passed the line yet area
    # missing (fillna(0)).
    n_t = (
        n_t.reindex(
            list(
                range(
                    traj_data.data.frame.min(), traj_data.data.frame.max() + 1
                )
            )
        )
        .ffill()
        .fillna(0)
    )

    n_t = n_t.to_frame()
    n_t.cumulative_pedestrians = n_t.cumulative_pedestrians.astype(int)

    # frame number is the index
    n_t[TIME_COL] = n_t.index / traj_data.frame_rate
    return n_t, crossing_frames


def compute_flow(
        *,
        nt: pd.DataFrame,
        crossing_frames: pd.DataFrame,
        individual_speed: pd.DataFrame,
        delta_frame: int,
        frame_rate: float,
) -> pd.DataFrame:
    r"""Compute the flow for the given the frame window from the nt information.

    Computes the flow :math:`J` in a frame interval of length
    :data:`delta_frame` (:math:`\Delta frame`). The first intervals starts,
    when the first person crossed the measurement, given by
    :data:`crossing_frames`. The next interval always starts at the time when
    the last person in the previous frame interval crossed the line.

    .. image:: /images/flow.svg
        :align: center
        :width: 80 %

    In each of the time interval it is checked, if any person has crossed the
    line, if yes, a flow $J$ can be computed. From the first frame the line was
    crossed :math:`f^{\Delta frame}_1`, the last frame someone crossed the line
    :math:`f^{\Delta frame}_N` the length of the frame interval
    :math:`\Delta f$` can be computed:

    .. math::

        \Delta f = f^{\Delta frame}_N - f^{\Delta frame}_1

    This directly together with the frame rate with :data:`frame_rate` ($fps$)
    gives the time interval $\Delta t$:

    .. math::

        \Delta t = \Delta f / fps

    Given the number of pedestrian crossing the line is given by
    :math:`N^{\Delta frame}`, the flow :math:`J` becomes:

    .. math::

        J = \frac{N^{\Delta frame}}{\Delta t}

    .. image:: /images/flow_zoom.svg
        :align: center
        :width: 60 %

    At the same time also the mean speed of the pedestrian when crossing the
    line is computed from :data:`individual_speed`.

    .. math::

        v_{crossing} = {1 \over N^{\Delta t} } \sum^{N^{\Delta t}}_{i=1} v_i(t)

    Args:
        nt (pd.DataFrame): DataFrame containing the columns 'frame',
            'cumulative_pedestrians', and 'time' (see result from
            :func:`~flow_calculator.compute_n_t`)
        crossing_frames (pd.DataFrame): DataFrame containing the columns
            'ID',  and 'frame' (see result from
            :func:`~flow_calculator.compute_n_t`)
        individual_speed (pd.DataFrame): DataFrame containing the columns
            'ID', 'frame', and 'speed'
        delta_frame (int): size of the frame interval to compute the flow
        frame_rate (float): frame rate of the trajectories

    Returns:
        DataFrame containing the columns 'flow' in 1/s, and 'mean_speed' in m/s.
    """
    crossing_speeds = pd.merge(
        crossing_frames, individual_speed, on=[ID_COL, FRAME_COL]
    )

    # Get frame where the first person passes the line
    num_passed_before = 0
    passed_frame_before = nt[nt[CUMULATED_COL] > 0].index.min()

    rows = []

    for frame in range(
            passed_frame_before + delta_frame, nt.index.max(), delta_frame
    ):
        passed_num_peds = nt.loc[frame][CUMULATED_COL]
        passed_frame = nt[nt[CUMULATED_COL] == passed_num_peds].index.min() + 1

        if passed_num_peds != num_passed_before:
            num_passing_peds = passed_num_peds - num_passed_before
            time_range = passed_frame - passed_frame_before

            flow_rate = num_passing_peds / time_range * frame_rate
            velocity = crossing_speeds[
                crossing_speeds.frame.between(
                    passed_frame_before, passed_frame, inclusive="both"
                )
            ][SPEED_COL].mean()

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame

            rows.append(
                {FLOW_COL: flow_rate, MEAN_SPEED_COL: velocity},
            )

    return pd.DataFrame(rows)


def calc_n(line):
    """calculates normal vector n of the line"""
    n = numpy.array([line.xy[1][1] - line.xy[1][0], line.xy[0][0] - line.xy[0][1]])
    return n / numpy.linalg.norm(n)


def partial_line_length(polygon, line):
    """calculates the fraction of the length that is intersected by the polygon"""
    return shapely.length(shapely.intersection(polygon, line)) / shapely.length(line)


def weight_value(group, n, line):
    """calculates the velocity in direction n times partial line length of the polygon
    group is a dataframe containing the columns 'v_x', 'v_y' and 'polygon' """
    return (group[V_X_COL] * n[0] + group[V_Y_COL] * n[1]) * partial_line_length(group[POLYGON_COL], line)


def merge_table(individual_voronoi_polygons, species, line, individual_speed=None):
    """merges the entries of the tables individual_voronoi_polygons, species, individual speed
    when a polygon intersects the line
    if no individual speed is given it does not merge the table
    """
    merged_table = individual_voronoi_polygons[shapely.intersects(individual_voronoi_polygons[POLYGON_COL], line)]
    merged_table = merged_table.merge(species, on="id", how="left")
    if individual_speed is None:
        return merged_table
    else:
        return merged_table.merge(individual_speed, left_on=[ID_COL, FRAME_COL], right_on=[ID_COL, FRAME_COL])


def separate_species(individual_voronoi_polygons, measurement_line: MeasurementLine, individual_speed):
    """creates a list containing the species for each agent"""
    # create dataframe with id and first frame where voronoi polygon intersects measurement line
    line = measurement_line.line
    intersecting_polys = individual_voronoi_polygons[shapely.intersects(individual_voronoi_polygons[POLYGON_COL], line)]
    min_idx = intersecting_polys.groupby(ID_COL)[FRAME_COL].idxmin()
    result = intersecting_polys.loc[min_idx, [ID_COL, FRAME_COL]]

    n = calc_n(line)

    # create dataframe with 'id' and 'species'
    end_result = result.merge(individual_speed, left_on=[ID_COL, FRAME_COL], right_on=[ID_COL, FRAME_COL])
    end_result[SPECIES_COL] = numpy.sign(n[0] * end_result[V_X_COL] + n[1] * end_result[V_Y_COL])
    return end_result[[ID_COL, SPECIES_COL]]


def calc_speed_on_line(individual_voronoi_polygons, measurement_line: MeasurementLine, individual_speed, species):
    # note:  species needs to contain the columns id, species where species == 1 or species == -1
    line = measurement_line.line
    n = calc_n(line)
    i_poly = merge_table(individual_voronoi_polygons, species, line, individual_speed)
    species_1 = i_poly[i_poly[SPECIES_COL] == 1]
    species_2 = i_poly[i_poly[SPECIES_COL] == -1]
    if not species_1.empty:
        species_1 = species_1.groupby(FRAME_COL).apply(
            lambda group: (weight_value(group, n, line)).sum()).reset_index()
        species_1.columns = [FRAME_COL, V_SP1_COL]
    else:
        species_1 = pd.DataFrame(columns=[FRAME_COL, V_SP1_COL])

    if not species_2.empty:
        species_2 = species_2.groupby(FRAME_COL).apply(
            lambda group: (weight_value(group, n, line)).sum()).reset_index()
        species_2.columns = [FRAME_COL, V_SP2_COL]
        species_2[V_SP2_COL] *= -1
    else:
        species_2 = pd.DataFrame(columns=[FRAME_COL, V_SP2_COL])

    result = species_1.merge(species_2, on=FRAME_COL, how="left")
    result[VELOCITY_COL] = result[V_SP1_COL] + result[V_SP2_COL]
    return result


def calc_density_on_line(individual_voronoi_polygons, measurement_line: MeasurementLine, species):
    # note:  species needs to contain the columns id, species where species == 1 or species == -1
    line = measurement_line.line
    i_poly = merge_table(individual_voronoi_polygons, species, line)
    species_1 = i_poly[i_poly[SPECIES_COL] == 1]
    species_2 = i_poly[i_poly[SPECIES_COL] == -1]
    if not species_1.empty:
        species_1 = species_1.groupby(FRAME_COL).apply(
            lambda group: (group[DENSITY_COL] * (partial_line_length(group[POLYGON_COL], line))).sum()).reset_index()
        species_1.columns = [FRAME_COL, DENSITY_SP1_COL]
    else:
        species_1 = pd.DataFrame(columns=[FRAME_COL, DENSITY_SP1_COL])

    if not species_2.empty:
        species_2 = species_2.groupby(FRAME_COL).apply(
            lambda group: (group[DENSITY_COL] * (partial_line_length(group[POLYGON_COL], line))).sum()).reset_index()

        species_2.columns = [FRAME_COL, DENSITY_SP2_COL]
    else:
        species_2 = pd.DataFrame(columns=[FRAME_COL, DENSITY_SP2_COL])

    result = species_1.merge(species_2, on=FRAME_COL, how="left")
    result[DENSITY_COL] = result[DENSITY_SP1_COL] + result[DENSITY_SP2_COL]
    return result


def calc_flow_on_line(individual_voronoi_polygons, measurement_line: MeasurementLine, individual_speed, species):
    # note:  species needs to contain the columns id, species where species == 1 or species == -1
    line = measurement_line.line
    n = calc_n(line)
    i_poly = merge_table(individual_voronoi_polygons, species, line, individual_speed)

    species_1 = i_poly[i_poly[SPECIES_COL] == 1]
    species_2 = i_poly[i_poly[SPECIES_COL] == -1]
    if not species_1.empty:
        species_1 = species_1.groupby(FRAME_COL).apply(
            lambda group: (weight_value(group, n, line) * group[DENSITY_COL]).sum()).reset_index()
        species_1.columns = [FRAME_COL, FLOW_SP1_COL]
    else:
        species_1 = pd.DataFrame(columns=[FRAME_COL, FLOW_SP1_COL])

    if not species_2.empty:
        species_2 = species_2.groupby(FRAME_COL).apply(
            lambda group: (weight_value(group, n, line) * group[DENSITY_COL]).sum()).reset_index()
        species_2.columns = [FRAME_COL, FLOW_SP2_COL]
        species_2[FLOW_SP2_COL] *= -1
    else:
        species_2 = pd.DataFrame(columns=[FRAME_COL, FLOW_SP2_COL])

    result = species_1.merge(species_2, on=FRAME_COL, how="left")
    result[FLOW_COL] = result[FLOW_SP1_COL] + result[FLOW_SP2_COL]
    return result
