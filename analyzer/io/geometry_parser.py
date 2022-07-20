"""This module provides the functionalities to parse geometries from the given input files.

Note:
    Currently only 2D geometries are supported!
"""

import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pygeos

from analyzer.data.geometry import Geometry


def parse_geometry(geometry_file: pathlib.Path) -> Geometry:
    """Parses the geometry from the given file.

    Args:
        geometry_file (pathlib.Path): file which should be parsed

    Returns:
        geometry object containing all relevant information

    """
    root = ET.parse(geometry_file).getroot()
    walls = _parse_geometry_walls(root)
    obstacles = _parse_obstacles(root)

    return Geometry(pygeos.polygonize(walls), obstacles)


def _parse_geometry_walls(xml_root: ET.Element) -> np.ndarray:
    """Parses all walls from the given root of the geometry file.

    Args:
        xml_root (ET.Element): root of the geometry file

    Returns:
        numpy array of all walls which are in the geometry
    """
    walls = []
    for polygon_node in xml_root.findall("rooms/room/subroom/polygon"):
        walls.append(_parse_vertex(polygon_node))
    return np.array(walls)


def _parse_obstacles(xml_root: ET.Element) -> np.ndarray:
    """Parses all obstacles from the given root of the geometry file.

    Args:
        xml_root (ET.Element): root of the geometry file

    Returns:
        numpy array of all obstacles (as polygons) which are in the geometry
    """
    obstacles = []
    for obstacles_node in xml_root.findall("rooms/room/subroom/obstacle/polygon"):
        obstacles.append(_parse_vertex(obstacles_node))
    return pygeos.get_parts(pygeos.normalize(pygeos.polygonize(np.array(obstacles))))


def _parse_vertex(polygon_node: ET.Element) -> pygeos.Geometry:
    """Parses the given vertex as line string

    Args:
        polygon_node (ET.Element): xml node containing the vertex information

    Returns:
        linestring of the parsed vertex
    """
    border = []
    for vertex in polygon_node.iter("vertex"):
        border.append([float(vertex.attrib["px"]), float(vertex.attrib["py"])])
    return pygeos.linestrings(np.array(border))
