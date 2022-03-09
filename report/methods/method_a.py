"""Computation of method A

This method calculates the mean value of flow and density over time.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from report.data.configuration import ConfigurationMethodA
from report.data.trajectory_data import TrajectoryData
from report.methods.velocity_calculator import VelocityCalculator


@dataclass(frozen=True)
class ResultMethodA:
    """Result of method A

    Attributes:
        df_nt (pd.DataFrame): data frame containing the frame number, time, and number of cumulative
                              pedestrians who passed the measurement line
        df_flow (pd.DataFrame): data ftame containing the flow, and mean_velocity for each
                                frame_interval someone passed the measurement line

    """

    df_nt: pd.DataFrame
    df_flow: pd.DataFrame


def run_method_a(
    configurations: Dict[int, ConfigurationMethodA],
    trajectory: TrajectoryData,
    measurement_lines: Dict[int, LineString],
    velocity_calculator: VelocityCalculator,
) -> Dict[int, ResultMethodA]:
    """Analyse the trajectories for each of the configurations and measurement lines.

    Arguments:
        configurations (Dict[int, ConfigurationMethodA]): configurations which should be used
                                                          for the analysis
        trajectory (TrajectoryData): trajectory to analyse
        measurement_lines (Dict[int, LineString]): lines which should be considered in the analysis
        velocity_calculator (VelocityCalculator): velocity calculator

    Returns:
        Dictionary containing the result of method A for each given measurement line (id of the
        measurement line is the key)
    """
    results = {}
    for measurement_line_id, configuration in configurations.items():
        results[measurement_line_id] = _run_method_a(
            configuration,
            trajectory,
            measurement_lines[measurement_line_id],
            velocity_calculator,
        )
    return results


def _run_method_a(
    configuration: ConfigurationMethodA,
    trajectory: TrajectoryData,
    measurement_line: LineString,
    velocity_calculator: VelocityCalculator,
) -> ResultMethodA:
    """Run method A for a single measurement line

    Arguments:
        configuration (ConfigurationMethodA): configuration which should be used for the analysis
        trajectory (TrajectoryData): trajectory to analyse
        measurement_line (LineString): line which should be considered in the analysis
        velocity_calculator (VelocityCalculator): velocity calculator

    Returns:
        Result of method A (see ResultMethodA for more detailed information)
    """
    passing_frames = _compute_passing_frames(trajectory, measurement_line)
    df_nt = _compute_n_t(trajectory, passing_frames)
    df_flow = _compute_flow(trajectory, configuration, passing_frames, velocity_calculator)
    return ResultMethodA(df_nt, df_flow)


def _compute_passing_frames(
    trajectory: TrajectoryData, measurement_line: LineString
) -> Dict[int, List[int]]:
    """Compute the frames and who passed the measurement_line

    Arguments:
        trajectory (TrajectoryData): trajectory to analyse
        measurement_line (LineString): line which should be considered in the analysis

    Returns:
        Dict at which frame pedestrians passed the line, with frame as key, and list of pedestrian
        ids as value
    """
    passing_frame = {}
    prev_positions = trajectory.get_positions(trajectory.get_min_frame())
    for frame in range(trajectory.get_min_frame() + 1, trajectory.get_max_frame() + 1):
        current_positions = trajectory.get_positions(frame)
        passing_agents = []
        for agent_id, current_position in current_positions.items():
            if agent_id in prev_positions.keys():
                step = LineString([prev_positions[agent_id], current_position])
                if step.intersection(measurement_line) and not measurement_line.contains(
                    prev_positions[agent_id]
                ):
                    passing_agents.append(agent_id)
        if passing_agents:
            passing_frame[frame] = passing_agents
        prev_positions = current_positions
    return passing_frame


def _compute_n_t(
    trajectory: TrajectoryData,
    passing_frames: Dict[int, List[int]],
) -> pd.DataFrame:
    """Compute the cumulative number of pedestrians who passed the line for each frame.

    Arguments:
        trajectory (TrajectoryData): trajectory to analyse
        passing_frames (Dict[int, List[int]]): Dict at which frame pedestrians passed the line (see
                                              _compute_passing_frames for more detail.

    Returns:
        DataFrame containing the frame, time, and cumulative number of pedestrians who passed
        the measurement line for each frame.
    """
    df_nt = pd.DataFrame(columns=["frame", "time", "num_peds"])
    passed_pedestrians = 0
    for frame in range(trajectory.get_min_frame(), trajectory.get_max_frame() + 1):
        current_time = frame / trajectory.frame_rate
        if frame in passing_frames:
            passed_pedestrians += len(passing_frames[frame])

        current_frame_nt = [frame, current_time, passed_pedestrians]
        df_nt.loc[frame] = current_frame_nt
    df_nt = df_nt.astype({"frame": int, "num_peds": int})

    return df_nt


def _compute_flow(
    trajectory: TrajectoryData,
    configuration: ConfigurationMethodA,
    passing_frames: Dict[int, List[int]],
    velocity_calculator: VelocityCalculator,
) -> pd.DataFrame:
    """Compute the flow and mean velocity.

    Arguments:
        trajectory (TrajectoryData): trajectory to analyse
        configuration (ConfigurationMethodA): configuration of the analysis
        passing_frames (Dict[int, List[int]]): Dict at which frame pedestrians passed the line (see
                                              _compute_passing_frames for more detail.
        velocity_calculator (VelocityCalculator): velocity calculator

    Returns:
        DataFrame containing the flow, and mean velocity for each time interval were pedestrians
        passed the measurement line.

    """
    df_flow = pd.DataFrame(columns=["flow", "mean_velocity"])
    passing = list(passing_frames.keys())
    first_passed_frame = passing[0]
    for min_frame in range(
        first_passed_frame, trajectory.get_max_frame(), configuration.frame_interval
    ):
        max_frame = min_frame + configuration.frame_interval

        passed_velocity = []
        passed_frames = [frame for frame in passing if min_frame <= frame < max_frame]

        for frame in passed_frames:
            for pedestrian in passing_frames[frame]:
                passed_velocity.append(
                    velocity_calculator.compute_instantaneous_velocity(
                        trajectory, pedestrian, frame
                    )
                )

        if not passed_velocity:
            continue

        # TODO bug in old jpsreport
        # - always missing one person when computing the flow
        # - min_time was always the last time someone passed the line (maybe not in time_interval)?
        num_passed_pedestrian = len(passed_velocity)
        min_time_passed = trajectory.get_time_in_s(min(passed_frames))
        max_time_passed = trajectory.get_time_in_s(max(passed_frames))

        flow = num_passed_pedestrian / np.abs(max_time_passed - min_time_passed)

        df_flow.loc[df_flow.shape[0]] = [flow, np.mean(passed_velocity)]

    return df_flow
