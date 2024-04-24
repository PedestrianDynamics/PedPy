#! /usr/bin/env python3

import argparse
import pathlib
import random
import sys
from typing import Any, List, TextIO

import numpy as np
import numpy.typing as npt
import pandas as pd

from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL


def required_length(nmin: int, nmax: int) -> Any:
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):  # type: ignore
            if not nmin <= len(values) <= nmax:
                msg = f"argument {self.dest} requires between {nmin} and {nmax} arguments"
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        help="file where the trajectory is written",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--fps",
        default=25,
        help="fps of the created trajectory (1/s)",
        type=float,
    )
    parser.add_argument(
        "--sim_time",
        default=80,
        help="length of the created trajectory (s)",
        type=float,
    )
    parser.add_argument(
        "--geometry",
        default="geometry.xml",
        help="place holder for the geometry file name",
        type=str,
    )
    parser.add_argument(
        "--random_ids",
        help="if used, the pedestrians will get a random positive ID",
        action="store_true",
    )
    parser.add_argument(
        "--movement_direction",
        help="direction each pedestrian is moving (default: (1.0, 0))",
        nargs="+",
        type=float,
        default=[1.0, 0.0],
        action=required_length(2, 2),
    )
    parser.add_argument(
        "--xrange",
        help="allowed range for the x coordinate, pedestrian will only be included in trajectory file, when inside. "
        "Usage --xrange <xmin> <xmax>",
        nargs="+",
        type=float,
        action=required_length(2, 2),
        default=[-10.0, 50.5],
    )
    parser.add_argument(
        "--yrange",
        help="allowed range for the y coordinate, pedestrian will only be included in trajectory file, when inside. "
        "Usage --yrange <ymin> <ymax>",
        nargs="+",
        type=float,
        action=required_length(2, 2),
    )
    parser.add_argument(
        "--velocity",
        type=float,
        help="velocity of the pedestrians (m/s)",
        default=1.0,
    )

    sub_parsers = parser.add_subparsers(
        help="sub-command help", required=True, dest="cmd"
    )

    grid_parser = sub_parsers.add_parser("grid")
    grid_parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[1, 1],
        action=required_length(2, 2),
    )
    grid_parser.add_argument(
        "--start_position",
        nargs="+",
        type=float,
        default=[-10, -4.0],
        action=required_length(2, 2),
    )

    grid_parser.add_argument("--ped_distance", type=float, default=1.0)

    return parser


def write_trajectory(
    out_file: pathlib.Path,
    fps: int,
    geometry: str,
    num_pedestrians: int,
    traj: pd.DataFrame,
) -> None:
    with open(out_file, "w") as trajectory_file:
        write_header(trajectory_file, fps, geometry, num_pedestrians)
        write_trajectory_data(trajectory_file, traj)


def write_header(
    trajectory_file: TextIO, fps: int, geometry_file: str, num_pedestrians: int
) -> None:
    version = "0.0.1"

    header = f"""#description: jpscore ({version})
#agents: {num_pedestrians}
#count: 0
#framerate: {fps}
#geometry: {geometry_file}
#ID: the agent ID
#FR: the current frame
#X,Y,Z: the agents coordinates (in metres)
#A, B: semi-axes of the ellipse
#ANGLE: orientation of the ellipse
#COLOR: color of the ellipse

#ID	FR	X	Y	Z	A	B	ANGLE	COLOR
"""

    trajectory_file.write(header)


def write_trajectory_data(
    trajectory_file: TextIO, trajectory_data: pd.DataFrame
) -> None:
    trajectory_data["Z"] = 0.0
    trajectory_data["A"] = 0.5
    trajectory_data["B"] = 0.5
    trajectory_data["ANGLE"] = 0.0
    trajectory_data["COLOR"] = 220

    trajectory_data.sort_values(["FR", "ID"], inplace=True, ascending=True)
    trajectory_file.write(
        trajectory_data.to_csv(sep="\t", header=False, index=False)
    )


def get_grid_trajectory(
    *,
    shape: List[int],
    start_position: npt.NDArray[np.float64],
    movement_direction: npt.NDArray[np.float64],
    ped_distance: float,
    random_ids: bool,
    number_frames: int,
) -> pd.DataFrame:
    """Generates a grid of pedestrians with the given shape.

    Args:
        shape (List[int]): Shape of the grid [#rows #cols]
        start_position (np.ndarray): starting position (bottom left of the grid) [x y]
        movement_direction (np.ndarray): movement direction
        ped_distance (float): distance between rows and columns of the grid
        random_ids (bool): whether to use positive random ids
        number_frames (int): number of frames to generate

    Returns:
        grid trajectory
    """
    number_peds = shape[0] * shape[1]

    traj = []
    if random_ids:
        ids = random.sample(range(500 * number_peds), number_peds)
        ids.sort()
    else:
        ids = list(range(number_peds))

    for i in range(shape[1]):
        for j in range(shape[0]):
            i_offset = i * np.array([1, 0])
            j_offset = j * np.array([0, 1])
            start = start_position + (i_offset + j_offset) * ped_distance

            ped_id = ids[i * shape[0] + j]

            for frame in range(int(number_frames)):
                position = start + frame * movement_direction
                traj.append([ped_id, frame, position[0], position[1]])

    return pd.DataFrame(traj, columns=[ID_COL, FRAME_COL, X_COL, Y_COL])


def filter_pedestrians(
    traj: pd.DataFrame, x_range: List[float], y_range: List[float]
) -> pd.DataFrame:
    """Filters the given trajectory, such that all the pedestrians are within the given range

    Args:
        traj (pd.DataFrame): trajectory data containing the unfiltered trajectories
        x_range (List[float]): range of allowed x-coordinates [x_min, x_max]
        y_range (List[float]): range of allowed y-coordinates [y_min, y_max]

    Returns:
        all the trajectory points within the given areas
    """
    filtered = traj
    if x_range is not None:
        filtered = filtered.loc[
            (x_range[0] <= filtered.X) & (filtered.X <= x_range[1])
        ]

    if y_range is not None:
        filtered = filtered.loc[
            (y_range[0] <= filtered.Y) & (filtered.Y <= y_range[1])
        ]

    return filtered


def get_movement_per_frame(
    direction: npt.NDArray[np.float64],
    speed: float,
    fps: float,
) -> npt.NDArray[np.float64]:
    """Compute the desired movement per frame

    Args:
        direction (np.ndarray): Movement direction [x y]
        speed (float): desired movement speed (m/s)
        fps (float): framerate of the trajectory file (1/s)

    Returns:
        desired movement per frame
    """
    direction = direction / np.linalg.norm(direction)
    return speed * direction / fps


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()

    num_frames = args.sim_time * args.fps
    movement = get_movement_per_frame(
        np.asarray(args.movement_direction), args.velocity, args.fps
    )

    if args.cmd == "grid":
        num_peds = args.shape[0] * args.shape[1]
        traj_data = get_grid_trajectory(
            shape=args.shape,
            start_position=np.asarray(args.start_position),
            ped_distance=args.ped_distance,
            random_ids=args.random_ids,
            number_frames=num_frames,
            movement_direction=movement,
        )
    else:
        print("Not supported subcommand-called")
        sys.exit(1)

    traj_data = filter_pedestrians(
        traj_data,
        x_range=args.xrange,
        y_range=args.yrange,
    )
    write_trajectory(args.output, args.fps, args.geometry, num_peds, traj_data)
