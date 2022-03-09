"""
This module provides the functionalities to parse the analysis configuration from the given
ini-file.
"""

import pathlib
from typing import Callable, Dict, List, Tuple, Union
from xml.etree.ElementTree import Element, parse

from shapely.geometry import LineString, Point

from report.data.configuration import Configuration, ConfigurationVelocity


class IniFileParseException(ValueError):
    """Custom exception for incomplete ini-files"""

    error_footer: str = (
        "For more detail check the documentation at: "
        "https://www.jupedsim.org/jpsreport_inifile.html.\n"
        "Please check your ini-file."
    )

    def __init__(self, message):
        self.message = f"{message}\n{self.error_footer}"
        super().__init__(self.message)


class IniFileValueException(ValueError):
    """Custom exception for wrong values inside the ini-file"""


def parse_xml_attrib(
    xml_element: Element,
    attrib: str,
    type_func: Callable[[str], Union[int, float, str, pathlib.Path, bool]],
    condition: Tuple[Callable[[Union[int, float, str, pathlib.Path, bool]], bool], str] = None,
    mandatory: bool = True,
) -> Union[int, float, str, pathlib.Path, bool, None]:
    """Parses a specific xml_element.attrib to the desired type and can also check if some
    condition is met.

    Args:
        xml_element (Element): xml Element to parse for the desired attrib
        attrib (str): name of the attrib to parse
        type_func (Callable[[str], Union[int, float, str, pathlib.Path, bool]]): cast operator to
            the desired data type
        condition (tuple[Callable[[Union[int, float, str, pathlib.Path, bool]], bool], str] = None):
            first element: condition which should be met for the output value,
            second element: error message to display then condition not met
        mandatory (bool = True): when not mandatory return None when element not found

    Returns:
        content of the xml_element.attrib in the desired datatype

    """
    if not callable(type_func):
        raise ValueError(f"{type_func} is not callable.")

    if attrib not in xml_element.attrib:
        if mandatory:
            raise IniFileParseException(
                f'Could not find "{attrib}"-attribute in "{xml_element.tag}"-tag, but is mandatory.'
            )
        return None

    try:
        result = type_func(xml_element.attrib[attrib])
    except ValueError as value_error:
        raise IniFileValueException(
            f'The "{attrib}"-attribute needs to be a {type_func.__name__} value, but is '
            f'"{xml_element.attrib[attrib]}". Please check your "{xml_element.tag}"-tag in '
            f"your ini-file."
        ) from value_error

    if condition is None:
        return result

    if not condition[0](result):
        raise IniFileValueException(condition[1] + f'("{xml_element.attrib[attrib]}")')

    return result


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
        raise IniFileParseException(
            "There could no output tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<output location="results"/>'
        )

    output_directory = parse_xml_attrib(
        output_directory_node,
        "location",
        pathlib.Path,
        (
            lambda x: x.exists() and x.is_dir(),
            "The output directory does not exist or is not a directory: ",
        ),
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
        raise IniFileParseException(
            "There could no geometry tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<geometry file="geometry.xml"/>'
        )

    geometry_file = parse_xml_attrib(
        geometry_node,
        "file",
        pathlib.Path,
        (
            lambda x: x.exists() and x.is_file(),
            "The geometry file does not exist or is not a file: ",
        ),
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
        raise IniFileParseException(
            "There could no trajectories tag be found in your ini-file, but it is mandatory."
        )

    if trajectory_node.attrib["format"]:
        traj_format = trajectory_node.attrib["format"]
        if traj_format != "txt":
            raise IniFileValueException("Only 'txt' is supported as trajectory file format. ")
    if len(trajectory_node.findall("file")) == 0:
        raise IniFileParseException(
            "There could no trajectories/name tag be found in your ini-file, but it is mandatory."
        )

    trajectory_files = []
    for file in trajectory_node.findall("file"):
        trajectory_file = parse_xml_attrib(
            file,
            "name",
            pathlib.Path,
        )
        trajectory_files.append(trajectory_file)

    # Check for missing trajectory files afterwards to report all missing files at once
    file_not_existing = []
    for file in trajectory_files:
        if not file.is_file():
            file_not_existing.append(file)

    if file_not_existing:
        raise IniFileValueException(
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
        raise IniFileParseException(
            "There could no measurement_areas tag be found in your ini-file, but it is mandatory."
        )

    for measurement in xml_root.iter("measurement_areas"):
        parse_xml_attrib(
            measurement,
            "unit",
            str,
            (
                lambda x: x.lower() == "m",
                "Only 'm' is supported as unit in the measurement areas tag: ",
            ),
            mandatory=False,
        )

        for measurement_line in measurement.iter("area_L"):
            line_id = parse_xml_attrib(
                measurement_line,
                "id",
                int,
                (
                    lambda x: x not in measurement_lines,
                    "There is a duplicated ID in your measurement lines: ",
                ),
            )

            start_node = measurement_line.find("start")
            if start_node is None:
                raise ValueError(
                    "The measurement_areas/area_L tag is incomplete, it should contain a start"
                    " child, e.g., \n"
                    '<area_L id="1">\n'
                    '    <start x="62.0" y="102.600"/>\n'
                    '    <end x="62.0" y="101.400"/>\n'
                    "</area_L>"
                )
            start_x = parse_xml_attrib(
                start_node,
                "x",
                float,
            )
            start_y = parse_xml_attrib(
                start_node,
                "y",
                float,
            )

            end_node = measurement_line.find("end")
            if end_node is None:
                raise IniFileParseException(
                    "The measurement_areas/area_L tag is incomplete, it should contain a end "
                    "child, e.g., \n"
                    '<area_L id="1">\n'
                    '    <start x="62.0" y="102.600"/>\n'
                    '    <end x="62.0" y="101.400"/>\n'
                    "</area_L>"
                )
            end_x = parse_xml_attrib(
                end_node,
                "x",
                float,
            )
            end_y = parse_xml_attrib(
                end_node,
                "y",
                float,
            )

            line = LineString([Point(start_x, start_y), Point(end_x, end_y)])
            if line.length <= 1e-5:
                raise IniFileValueException(
                    f"The measurement line {id} is too narrow. Check your start and end point."
                    f"Distance between start and end is {line.length}."
                    "Please check your ini-file."
                )

            measurement_lines[line_id] = line
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
        raise IniFileParseException(
            "There could no velocity tag be found in your ini-file, but it is "
            "mandatory, e.g.,  \n"
            '<velocity frame_step="10" set_movement_direction="None" '
            'ignore_backward_movement="false"/>'
        )

    frame_step = parse_xml_attrib(
        velocity_root,
        "frame_step",
        int,
        (
            lambda x: x > 0,
            "The velocity frame_step needs to be a positive integer value, but is: ",
        ),
    )

    set_movement_direction = parse_xml_attrib(
        velocity_root,
        "set_movement_direction",
        str,
    )

    ignore_backward_movement_str = parse_xml_attrib(
        velocity_root,
        "ignore_backward_movement",
        str,
    ).lower()

    if "true" not in ignore_backward_movement_str and "false" not in ignore_backward_movement_str:
        raise IniFileValueException(
            f"The velocity ignore_backward_movement needs to be a boolean value ('True', 'False'),"
            f" but is {ignore_backward_movement_str}. "
            f"Please check your velocity tag in your ini-file."
        )
    ignore_backward_movement = ignore_backward_movement_str == "true"

    return ConfigurationVelocity(frame_step, set_movement_direction, ignore_backward_movement)
