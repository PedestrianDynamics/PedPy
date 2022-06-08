import pathlib
import xml.etree.ElementTree
from typing import Dict, List

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from report.data.configuration import ConfigurationVelocity
from report.io.ini_parser import (
    parse_geometry_file,
    parse_measurement_lines,
    parse_output_directory,
    parse_trajectory_files,
    parse_velocity_configuration,
)
from report.version import __version__


def get_ini_file_as_string(content: str):
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<JPSreport project="JPS-Project" version="{__version__}">\n'
        f"{content}\n"
        f"</JPSreport>\n"
    )


def test_parse_output_directory_success(tmp_path):
    output_directory = tmp_path / "results"
    output_directory.mkdir()

    geometry_xml = f'<output location="{output_directory}"/>\n'
    xml_content = get_ini_file_as_string(geometry_xml)
    root = xml.etree.ElementTree.fromstring(xml_content)

    output_directory_from_file = parse_output_directory(root)
    assert output_directory_from_file == output_directory


def test_parse_output_directory_directory_does_not_exist():
    folder_name = "results"
    xml_content = get_ini_file_as_string(f'<output location="{folder_name}"/>')
    root = xml.etree.ElementTree.fromstring(xml_content)

    output_directory_from_file = parse_output_directory(root)
    assert folder_name == output_directory_from_file.name


def test_parse_output_directory_directory_is_not_a_directory(tmp_path):
    output_directory = tmp_path / "not_a_directory.txt"
    output_directory.touch()
    xml_content = get_ini_file_as_string(f'<output location="{output_directory}"/>')
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_output_directory(root)
    assert "The output directory does exist but is not a directory:" in str(error_info.value)


@pytest.mark.parametrize(
    "content, expected_message",
    [
        ("", "There could no output tag be found in your ini-file,"),
        (
            '<not_output location="results"/>',
            "There could no output tag be found in your ini-file,",
        ),
        (
            "<output/>",
            'Could not find "location"-attribute in "output"-tag',
        ),
        (
            '<output not_location="results"/>',
            'Could not find "location"-attribute in "output"-tag',
        ),
    ],
)
def test_parse_output_directory_wrong_input(tmp_path, content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_output_directory(root)
    assert expected_message in str(error_info.value)


def test_parse_geometry_file_success(tmp_path):
    geometry_file = tmp_path / "geometry.xml"
    geometry_file.touch()

    geometry_xml = f'<geometry file="{geometry_file}"/>\n'
    xml_content = get_ini_file_as_string(geometry_xml)
    root = xml.etree.ElementTree.fromstring(xml_content)

    geometry_from_file = parse_geometry_file(root)
    assert geometry_from_file == geometry_file


@pytest.mark.parametrize(
    "content, expected_message",
    [
        (
            '<geometry file="geo.xml"/>',
            "The geometry file does not exist or is not a file:",
        ),
    ],
)
def test_parse_geometry_file_wrong_data(tmp_path, content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_geometry_file(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "content, expected_message",
    [
        ("", "There could no geometry tag be found in your ini-file,"),
        (
            '<not_geometry file="geo.xml"/>',
            "There could no geometry tag be found in your ini-file,",
        ),
        (
            "<geometry/>",
            'Could not find "file"-attribute in "geometry"-tag,',
        ),
        (
            '<geometry not_file="geo.xml"/>',
            'Could not find "file"-attribute in "geometry"-tag,',
        ),
    ],
)
def test_parse_geometry_file_wrong_input(tmp_path, content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_geometry_file(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "files",
    [
        ["traj_1.txt"],
        ["traj_1.txt", "traj_2.txt", "traj_2.txt"],
    ],
)
def test_parse_trajectory_files_success(tmp_path, files: List[pathlib.Path]):
    files_xml = '<trajectories format="txt">\n'
    expected_files = list()

    for traj_file in files:
        file = tmp_path / traj_file
        file.touch()
        files_xml += f'\t<file name="{file}"/>'

        expected_files.append(file)
    files_xml += "</trajectories>"

    xml_content = get_ini_file_as_string(files_xml)
    root = xml.etree.ElementTree.fromstring(xml_content)
    traj_from_file = parse_trajectory_files(root)

    assert traj_from_file == expected_files


@pytest.mark.parametrize(
    "content, expected_message",
    [
        (
            '<trajectories format="txt">\n' '\t<file name="traj.txt"/>\n' "</trajectories>\n",
            "The following trajectory files do not exist:",
        ),
    ],
)
def test_parse_trajectory_files_wrong_data(tmp_path, content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_trajectory_files(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "content, expected_message",
    [
        ("", "There could no trajectories tag be found in your ini-file,"),
        ('<trajectories format="txt"/>\n', "There could no trajectories/name tag be found in your"),
        (
            '<trajectories format="csv"/>\n',
            "Only 'txt' is supported as trajectory file format.",
        ),
        (
            '<trajectories format="txt">\n' '\t<not_file name="traj.txt"/>\n' "</trajectories>\n",
            "There could no trajectories/name tag be found in your",
        ),
        (
            '<trajectories format="txt">\n' '\t<file not_name="traj.txt"/>\n' "</trajectories>\n",
            'Could not find "name"-attribute in "file"-tag,',
        ),
    ],
)
def test_parse_trajectory_files_wrong_input(tmp_path, content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        file = tmp_path / "traj.txt"
        file.touch()
        parse_trajectory_files(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "lines, split_in_tags",
    [
        ({}, False),
        ({0: LineString([Point(0, 1), Point(2, 3)])}, False),
        (
            {
                0: LineString([Point(0, 1), Point(2, 3)]),
                1: LineString([Point(0, -1), Point(-2, -3)]),
            },
            False,
        ),
        (
            {
                0: LineString([Point(0, 1), Point(2, 3)]),
                1: LineString([Point(0, -1), Point(-2, -3)]),
            },
            True,
        ),
        (
            {
                0: LineString([Point(0, 1), Point(2, 3)]),
                1: LineString([Point(0, -1), Point(-2, -3)]),
                -1: LineString([Point(0, -1), Point(-2, -3)]),
            },
            True,
        ),
    ],
)
def test_parse_measurement_lines_success(lines: Dict[int, LineString], split_in_tags: bool):
    measurement_lines_xml = ""

    if not split_in_tags:
        measurement_lines_xml += "<measurement_areas>\n"
    for line_id, line in lines.items():
        if split_in_tags:
            measurement_lines_xml += "<measurement_areas>\n"

        measurement_lines_xml += f'\t<area_L id="{line_id}">\n'
        if line_id < 0:
            measurement_lines_xml += f'\t\t<end x="{line.xy[0][1]}" y="{line.coords.xy[1][1]}"/>\n'
            measurement_lines_xml += (
                f'\t\t<start x="{line.xy[0][0]}" y="{line.coords.xy[1][0]}"/>\n'
            )
        else:
            measurement_lines_xml += (
                f'\t\t<start x="{line.xy[0][0]}" y="{line.coords.xy[1][0]}"/>\n'
            )
            measurement_lines_xml += f'\t\t<end x="{line.xy[0][1]}" y="{line.coords.xy[1][1]}"/>\n'

        measurement_lines_xml += "\t</area_L>\n"
        if split_in_tags:
            measurement_lines_xml += "</measurement_areas>\n"

    if not split_in_tags:
        measurement_lines_xml += "</measurement_areas>\n"

    xml_content = get_ini_file_as_string(measurement_lines_xml)
    root = xml.etree.ElementTree.fromstring(xml_content)

    measurement_lines_from_file = parse_measurement_lines(root)

    assert measurement_lines_from_file == lines


@pytest.mark.parametrize(
    "content, expected_message",
    [
        (
            "<measurement_areas>\n"
            '\t<area_L id="not an integer">\n'
            '\t\t<start x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "id"-attribute needs to be a int value,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1.">\n'
            '\t\t<start x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "id"-attribute needs to be a int value,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="0">\n'
            '\t\t<start x="not a float" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "x"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="1" y="not a float"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "y"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="-1">\n'
            '\t\t<start x="not a float" y="not a float"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "x"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="0">\n'
            '\t\t<end x="not a float" y="-1.0"/>\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "x"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<end x="1" y="not a float"/>\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "y"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="-1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="not a float" y="not a float"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "x"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="-1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="not a float" y="not a float"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'The "x"-attribute needs to be a float value',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="7."/>\n'
            "\t</area_L>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="7."/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "There is a duplicated ID",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="0">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="7."/>\n'
            "\t</area_L>\n"
            '\t<area_L id="0">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="7."/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "There is a duplicated ID ",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="-5">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="8."/>\n'
            "\t</area_L>\n"
            '\t<area_L id="-5">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="1" y="7."/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "There is a duplicated ID ",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="-1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "is too narrow",
        ),
    ],
)
def test_parse_measurement_lines_wrong_data(content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_measurement_lines(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "content, expected_message",
    [
        (" ", "There could no measurement_areas tag be found in your ini-file,"),
        (
            '<measurement_areas unit="cm">\n'
            '\t<area_L id="1">\n'
            '\t\t<start x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "Only 'm' is supported as unit in the measurement areas tag:",
        ),
        (
            "<measurement_areas>\n"
            "\t<area_L>\n"
            '\t\t<start x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "id"-attribute in "area_L"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<not_start x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "The measurement_areas/area_L tag is incomplete, it should contain a start child,",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "The measurement_areas/area_L tag is incomplete, it should contain a start child,",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="0.0" y="-1.0"/>\n'
            '\t\t<not_end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "The measurement_areas/area_L tag is incomplete, it should contain a end child,",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="0.0" y="-1.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            "The measurement_areas/area_L tag is incomplete, it should contain a end child,",
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start not_x="0.0" y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "start"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start  y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "start"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="0.0" not_y="-1.0"/>\n'
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "y"-attribute in "start"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            "\t\t<start/>\n"
            '\t\t<end x="-2.0" y="-3.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "start"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end not_x="0.0" y="-1.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "end"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end y="-1.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "end"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            '\t\t<end x="0.0" not_y="-1.0"/>\n'
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "y"-attribute in "end"-tag,',
        ),
        (
            "<measurement_areas>\n"
            '\t<area_L id="1">\n'
            '\t\t<start x="-2.0" y="-3.0"/>\n'
            "\t\t<end/>\n"
            "\t</area_L>\n"
            "</measurement_areas>",
            'Could not find "x"-attribute in "end"-tag,',
        ),
    ],
)
def test_parse_measurement_lines_wrong_input(content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_measurement_lines(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "frame_step, movement_direction",
    [
        (10, "(0, 1)"),
        (50, "(1, 1)"),
        (1, "None"),
        (6, "(-1, -10)"),
    ],
)
def test_parse_velocity_configuration_success(frame_step, movement_direction):
    velocity_configuration = (
        f'<velocity frame_step="{frame_step}" ' f'movement_direction="{movement_direction}"/>'
    )

    xml_content = get_ini_file_as_string(velocity_configuration)
    root = xml.etree.ElementTree.fromstring(xml_content)

    movement_direction_array = None
    if movement_direction != "None":
        movement_direction_str = movement_direction.replace("(", "").replace(")", "")
        movement_direction_array = np.fromstring(movement_direction_str, dtype=float, sep=",")

    expected_velocity_configuration = ConfigurationVelocity(frame_step, movement_direction_array)

    velocity_configuration_from_file = parse_velocity_configuration(root)
    assert velocity_configuration_from_file.frame_step == expected_velocity_configuration.frame_step
    if movement_direction_array is not None:
        assert (
            velocity_configuration_from_file.movement_direction
            == expected_velocity_configuration.movement_direction
        ).all()
    else:
        assert (
            velocity_configuration_from_file.movement_direction
            == expected_velocity_configuration.movement_direction
        )


@pytest.mark.parametrize(
    "frame_step, movement_direction, expected_message",
    [
        (0, "None", "The velocity frame_step needs to be a positive integer value, "),
        (-50, "(1, 0)", "The velocity frame_step needs to be a positive integer value, "),
        (
            3.0,
            "(0, 5)",
            'The "frame_step"-attribute needs to be a int value,',
        ),
        (
            "not a integer point value",
            "(1, 0)",
            'The "frame_step"-attribute needs to be a int value,',
        ),
        (
            1,
            "",
            "The velocity movement_direction needs to be a 2 element sized vector with",
        ),
        (
            6,
            "foo",
            "The velocity movement_direction needs to be a 2 element sized vector with",
        ),
    ],
)
def test_parse_velocity_parser_wrong_data(frame_step, movement_direction, expected_message):
    velocity_calculator = (
        f'<velocity frame_step="{frame_step}" ' f'movement_direction="{movement_direction}"/>'
    )

    xml_content = get_ini_file_as_string(velocity_calculator)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_velocity_configuration(root)
    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "content, expected_message",
    [
        (" ", "There could no velocity tag be found in your ini-file,"),
        ("<velocity/>", 'Could not find "frame_step"-attribute in "velocity"-tag,'),
        (
            '<velocity movement_direction="None"/>',
            'Could not find "frame_step"-attribute in "velocity"-tag,',
        ),
        (
            '<velocity movement_direction="None"/>',
            'Could not find "frame_step"-attribute in "velocity"-tag,',
        ),
        (
            '<velocity ignore_backward_movement="false" movement_direction="None"/>',
            'Could not find "frame_step"-attribute in "velocity"-tag,',
        ),
    ],
)
def test_parse_velocity_parser_wrong_input(content, expected_message):
    xml_content = get_ini_file_as_string(content)
    root = xml.etree.ElementTree.fromstring(xml_content)

    with pytest.raises(ValueError) as error_info:
        parse_velocity_configuration(root)
    assert expected_message in str(error_info.value)
