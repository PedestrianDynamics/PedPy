"""This module provides the functionalities to parse the analysis configuration from the given
ini-file.
"""

import pathlib
from typing import Dict, List
from xml.etree.ElementTree import Element, parse

from shapely.geometry import LineString, Point

from report.data.configuration import Configuration, ConfigurationVelocity


def parse_ini_file(ini_file: pathlib.Path) -> Configuration:
    """Parses the configuration from the given ini-file

    Args:
        ini_file (pathlib.Path):  file which should be parsed

    Returns:
        configuration object containing all the needed configuration for the analysis
    """
    root = parse(ini_file).getroot()

    output_directory = parse_output_directory(root)
    trajectory_files = parse_trajectory_files(root)
    geometry_file = parse_geometry_file(root)
    measurement_areas = {}
    measurement_lines = parse_measurement_lines(root)
    velocity_configuration = parse_velocity_configuration(root)

    return Configuration(
        output_directory=output_directory,
        trajectory_files=trajectory_files,
        geometry_file=geometry_file,
        measurement_areas=measurement_areas,
        measurement_lines=measurement_lines,
        velocity_configuration=velocity_configuration,
    )


def parse_output_directory(xml_root: Element) -> pathlib.Path:
    """
    Args:
        xml_root (ET.ElementTree):  root of the xml file

    Returns:
        the output directory for the results
    """
    output_directory_node = xml_root.find("output")
    if output_directory_node is None:
        raise ValueError(
            "There could no output tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<output location="results"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )
    if "location" not in output_directory_node.attrib:
        raise ValueError(
            "The output tag is incomplete, it should contain 'location', e.g., \n"
            '<output location="results"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )
    output_directory = pathlib.Path(output_directory_node.attrib["location"])
    if not output_directory.is_dir():
        raise ValueError(
            f"The output directory does not exist or is not a directory: "
            f"{output_directory.__str__()}. "
            f"Please check your ini-file."
        )

    return output_directory


def parse_geometry_file(xml_root: Element) -> pathlib.Path:
    """
    Args:
        xml_root (ET.ElementTree):  root of the xml file

    Returns:
        path to the geometry file to use in the analysis
    """
    geometry_node = xml_root.find("geometry")
    if geometry_node is None:
        raise ValueError(
            "There could no geometry tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<geometry file="geometry.xml"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )
    if "file" not in geometry_node.attrib:
        raise ValueError(
            "The geometry tag is incomplete, it should contain 'name', e.g., \n"
            '<geometry file="geometry.xml"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )
    geometry_file = pathlib.Path(geometry_node.attrib["file"])
    if not geometry_file.is_file():
        raise ValueError(
            f"The geometry file does not exist: {geometry_file.__str__()}. "
            f"Please check your ini-file."
        )

    return geometry_file


def parse_trajectory_files(xml_root: Element) -> List[pathlib.Path]:
    """
    Args:
        xml_root (ET.ElementTree):  root of the xml file

    Returns:
        list of the trajectory files to use in the analysis
    """

    trajectory_node = xml_root.find("trajectories")
    if trajectory_node is None:
        raise ValueError(
            "There could no trajectories tag be found in your ini-file, but it is "
            "mandatory. For more detailed check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )

    if trajectory_node.attrib["format"]:
        traj_format = trajectory_node.attrib["format"]
        if traj_format != "txt":
            raise ValueError(
                "Only 'txt' is supported as trajectory file format. "
                "For more detailed check the documentation at: "
                "https://www.jupedsim.org/jpsreport_inifile.html\n"
                "Please check your ini-file."
            )
    if len(trajectory_node.findall("file")) == 0:
        raise ValueError(
            "There could no trajectories/name tag be found in your ini-file, but it is "
            "mandatory. For more detailed check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )

    trajectory_files = []
    for file in trajectory_node.findall("file"):
        if "name" in file.attrib:
            trajectory_files.append(pathlib.Path(file.attrib["name"]))
        else:
            raise ValueError(
                "There seems to be a trajectories/name tag be missing in your ini-file, but it is "
                "mandatory. For more detailed check the documentation at: "
                "https://www.jupedsim.org/jpsreport_inifile.html\n"
                "Please check your ini-file."
            )

    file_not_existing = []
    for file in trajectory_files:
        if not file.is_file():
            file_not_existing.append(file)

    if file_not_existing:
        raise ValueError(
            f"The following trajectory files do not exist: "
            f"{[file.__str__() for file in file_not_existing]}. "
            f"Please check your ini-file."
        )
    return trajectory_files


def parse_measurement_lines(xml_root: Element) -> Dict[int, LineString]:
    """Parses the measurement line from the given xml root
    Args:
        xml_root (ET.ElementTree):  root of the xml file

    Returns:
        measurement lines to use in the analysis (id, line)
    """
    measurement_lines = {}

    if xml_root.find("measurement_areas") is None:
        raise ValueError(
            "There could no measurement_areas tag be found in your ini-file, but it is "
            "mandatory. For more detailed check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )

    for measurement in xml_root.iter("measurement_areas"):
        if "unit" in measurement.attrib.keys():
            unit = measurement.attrib["unit"]
            if unit.lower() != "m":
                raise ValueError(
                    "Only 'm' is supported as unit in the measurement areas tag. "
                    "For more detailed check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )
        for measurement_line in measurement.iter("area_L"):
            needed_tags = {"id"}
            point_needed_tags = {"x", "y"}
            if not needed_tags.issubset(measurement_line.attrib.keys()):
                needed_tags_text = ", ".join(str(e) for e in needed_tags)
                raise ValueError(
                    f"The measurement_areas/area_L tag is incomplete, it should contain "
                    f"{needed_tags_text}, e.g., \n"
                    '<area_L id="1">\n'
                    '    <start x="62.0" y="102.600"/>\n'
                    '    <end x="62.0" y="101.400"/>\n'
                    "</area_L>\n"
                    "For more detail check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )

            try:
                line_id = int(measurement_line.attrib["id"])
            except ValueError:
                line_id = measurement_line.attrib["id"]
                raise ValueError(
                    f"The measurement_areas/area_L id needs to be a integer value, "
                    f"but is {line_id}. "
                    f"Please check your velocity tag in your ini-file."
                )

            start_node = measurement_line.find("start")
            if start_node is None:
                raise ValueError(
                    "The measurement_areas/area_L tag is incomplete, it should contain a start"
                    " child, e.g., \n"
                    '<area_L id="1">\n'
                    '    <start x="62.0" y="102.600"/>\n'
                    '    <end x="62.0" y="101.400"/>\n'
                    "</area_L>\n"
                    "For more detail check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )
            if not point_needed_tags.issubset(start_node.attrib.keys()):
                point_needed_tags_text = ", ".join(str(e) for e in point_needed_tags)
                raise ValueError(
                    f"The measurement_areas/area_L/start tag is incomplete, it should contain "
                    f"{point_needed_tags_text}, e.g., \n"
                    '<start x="62.0" y="102.600"/>\n'
                    "For more detail check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )

            try:
                start_x = float(start_node.attrib["x"])
                start_y = float(start_node.attrib["y"])
            except ValueError:
                start_x = start_node.attrib["x"]
                start_y = start_node.attrib["y"]
                raise ValueError(
                    f"The start point coordinates of measurement line {id} need to be floating "
                    f"point values, but are {start_x} and {start_y}. "
                    f"Please check your velocity tag in your ini-file."
                )

            end_node = measurement_line.find("end")
            if end_node is None:
                raise ValueError(
                    "The measurement_areas/area_L tag is incomplete, it should contain a end "
                    "child, e.g., \n"
                    '<area_L id="1">\n'
                    '    <start x="62.0" y="102.600"/>\n'
                    '    <end x="62.0" y="101.400"/>\n'
                    "</area_L>\n"
                    "For more detail check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )

            if not point_needed_tags.issubset(end_node.attrib.keys()):
                point_needed_tags_text = ", ".join(str(e) for e in point_needed_tags)
                raise ValueError(
                    f"The measurement_areas/area_L/end tag is incomplete, it should contain "
                    f"{point_needed_tags_text}, e.g., \n"
                    '<end x="62.0" y="101.400"/>\n'
                    "For more detail check the documentation at: "
                    "https://www.jupedsim.org/jpsreport_inifile.html\n"
                    "Please check your ini-file."
                )
            try:
                end_x = float(end_node.attrib["x"])
                end_y = float(end_node.attrib["y"])
            except ValueError:
                end_x = end_node.attrib["x"]
                end_y = end_node.attrib["y"]
                raise ValueError(
                    f"The end point coordinates of measurement line {id} need to be floating "
                    f"point values, but are {end_x} and {end_y}. "
                    f"Please check your velocity tag in your ini-file."
                )

            line = LineString([Point(start_x, start_y), Point(end_x, end_y)])
            if line.length <= 1e-5:
                raise ValueError(
                    f"The measurement line {id} is too narrow. Check your start and end point."
                    f"Distance between start and end is {line.length}."
                    "Please check your ini-file."
                )

            if line_id not in measurement_lines:
                measurement_lines[line_id] = line
            else:
                raise ValueError(
                    f"There is a duplicated id ({line_id}) in your measurement lines. "
                    "Please check your ini-file."
                )
    return measurement_lines


def parse_velocity_configuration(xml_root: Element) -> ConfigurationVelocity:
    """Parses the configuration for velocity computation from the given xml root

    Args:
        xml_root (ET.ElementTree):  root of the xml file

    Returns:
        configuration for the velocity computation to use in the analysis
    """
    velocity_root = xml_root.find("velocity")
    if velocity_root is None:
        raise ValueError(
            "There could no velocity tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<velocity frame_step="10" set_movement_direction="None" '
            'ignore_backward_movement="false"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )

    needed_tags = {"frame_step", "set_movement_direction", "ignore_backward_movement"}
    if not needed_tags.issubset(velocity_root.attrib.keys()):
        needed_tags_text = ", ".join(str(e) for e in needed_tags)
        raise ValueError(
            f"The velocity tag is incomplete, it should contain {needed_tags_text}, e.g., \n"
            '<velocity frame_step="10" set_movement_direction="None" '
            'ignore_backward_movement="false"/>\n'
            "For more detail check the documentation at: "
            "https://www.jupedsim.org/jpsreport_inifile.html\n"
            "Please check your ini-file."
        )

    try:
        frame_step = int(velocity_root.attrib["frame_step"])
    except ValueError:
        frame_step = velocity_root.attrib["frame_step"]
        raise ValueError(
            f"The velocity frame_step needs to be a positive integer value, but is {frame_step}. "
            f"Please check your velocity tag in your ini-file."
        )

    if frame_step <= 0:
        raise ValueError(
            f"The velocity frame_step needs to be a positive integer value, but is {frame_step}. "
            f"Please check your velocity tag in your ini-file."
        )

    set_movement_direction = velocity_root.attrib["set_movement_direction"]

    ignore_backward_movement_lower = velocity_root.attrib["ignore_backward_movement"].lower()
    if (
        "true" not in ignore_backward_movement_lower
        and "false" not in ignore_backward_movement_lower
    ):
        raise ValueError(
            f"The velocity ignore_backward_movement needs to be a boolean value ('True', 'False'),"
            f" but is {ignore_backward_movement_lower}. "
            f"Please check your velocity tag in your ini-file."
        )
    ignore_backward_movement = velocity_root.attrib["ignore_backward_movement"].lower() == "true"

    return ConfigurationVelocity(frame_step, set_movement_direction, ignore_backward_movement)
