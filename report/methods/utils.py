from typing import Dict, List

from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree

from report.data.trajectory_data import TrajectoryData


def compute_passing_in_frame(trajectory: TrajectoryData, measurement_line: LineString, frame: int):
    if frame <= trajectory.get_min_frame() or frame > trajectory.get_max_frame():
        raise RuntimeError("frame out of bounds")

    prev_positions = trajectory.get_positions(frame - 1)
    current_positions = trajectory.get_positions(frame)

    passing_agents = []
    for agent_id, current_position in current_positions.items():
        if agent_id in prev_positions.keys():
            step = LineString([prev_positions[agent_id], current_position])

            if step.intersection(measurement_line) and not measurement_line.contains(
                prev_positions[agent_id]
            ):
                passing_agents.append(agent_id)

    return passing_agents


def compute_passing_frames(
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


def compute_voronoi_polygons(pedestrian_positions: List[Point], border: Polygon) -> List[Polygon]:
    points = MultiPoint(pedestrian_positions)

    regions = STRtree(voronoi_diagram(points, envelope=border).geoms)
    ordered = GeometryCollection([regions.nearest(point) for point in points.geoms])
    ordered = GeometryCollection([border.intersection(p) for p in ordered.geoms])
    return ordered
