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

    return Geometry(pygeos.polygonize(walls))


def _parse_geometry_walls(xml_root: ET.Element) -> np.ndarray:
    """Parses the walls from the given xml node.

    Args:
        xml_root (ET.Element): root of the xml file

    Returns:
        list of all walls which are in the geometry
    """
    walls = []
    for subroom in xml_root.iter("subroom"):
        for polygon in subroom.iter("polygon"):
            wall = []
            for vertex in polygon.iter("vertex"):
                x = float(vertex.attrib["px"])
                y = float(vertex.attrib["py"])
                wall.append([x, y])
            walls.append(pygeos.linestrings(np.array(wall)))
    return np.array(walls)
