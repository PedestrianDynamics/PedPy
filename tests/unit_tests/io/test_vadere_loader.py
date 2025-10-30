import json
import math
import pathlib
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import shapely
from shapely import LinearRing, Polygon
from tests.unit_tests.io.utils import (
    get_data_frame_to_write,
    prepare_data_frame,
)

from pedpy import FRAME_COL, ID_COL, X_COL, Y_COL, LoadTrajectoryError
from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import TrajectoryUnit
from pedpy.io.vadere_loader import load_trajectory_from_vadere, load_walkable_area_from_vadere_scenario
from pedpy.io.viswalk_loader import load_trajectory_from_viswalk


def write_vadere_csv_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: float,
):
    data = data.rename(
        columns={
            ID_COL: "pedestrianId",
            FRAME_COL: "simTime",
            X_COL: "startX-PID1",  # "-PID1" stands for processor id 1 and is used in Vadere
            # outputs as extension of the generic column name startX
            Y_COL: "startY-PID1",  # "-PID1" see comment above
        }
    )
    data["simTime"] = data["simTime"] / frame_rate

    vadere_traj_header = "#IDXCOL=2,DATACOL=2,SEP=' '\n"
    with open(file, "w", encoding="utf-8-sig") as writer:
        writer.write(vadere_traj_header)

    data.to_csv(file, sep=" ", index=False, mode="a", encoding="utf-8-sig")


def write_vadere_scenario_file(
    file,
    complete_area,
    obstacles,
    bounding_box_width,
):
    obstacles_ = list()

    for obstacle in obstacles:
        obst_coords = list(obstacle.coords)

        if is_rectangular(shapely.Polygon(obstacle)):
            minx, miny, maxx, maxy = shapely.Polygon(obstacle).bounds
            obstacles_ += [
                {
                    "shape": {
                        "type": "RECTANGLE",
                        "x": obst_coords[0][0],
                        "y": obst_coords[0][1],
                        "width": abs(maxx - minx),
                        "height": abs(maxy - miny),
                    }
                }
            ]
        else:  # is polygon
            obstacles_ += [
                {
                    "shape": {
                        "type": "POLYGON",
                        "points": [{"x": p[0], "y": p[1]} for p in obst_coords],
                    }
                }
            ]

    if is_rectangular(shapely.Polygon(complete_area)):
        scenario = {
            "name": "vadere_test",
            "release": str(),
            "scenario": {
                "topography": {
                    "attributes": {
                        "bounds": {
                            "x": complete_area.bounds[0],
                            "y": complete_area.bounds[1],
                            "width": abs(complete_area.bounds[2] - complete_area.bounds[0]),
                            "height": abs(complete_area.bounds[3] - complete_area.bounds[1]),
                        },
                        "boundingBoxWidth": bounding_box_width,
                    },
                    "obstacles": obstacles_,
                }
            },
        }
    else:
        raise RuntimeError("Internal Error: Trying to write non-rectangular shape as Vadere scenario bound.")

    # Convert and write JSON object to file
    with open(file, "w") as f:
        json.dump(scenario, f, indent=2)


def is_rectangular(poly: shapely.Polygon):
    return math.isclose(
        a=poly.area,
        b=poly.minimum_rotated_rectangle.area,
        abs_tol=0,
    )


def force_cw(polygon):
    if is_ccw(polygon):
        return Polygon(polygon.exterior.coords[::-1])
    else:
        return polygon


def is_ccw(polygon):
    return LinearRing(polygon.exterior.coords).is_ccw


def test_load_trajectory_from_vadere_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/vadere_postvis.traj")
    load_trajectory_from_vadere(trajectory_file=traj_txt, frame_rate=24.0)


def test_load_trajectory_from_vadere_no_data(
    tmp_path: pathlib.Path,
):
    data_empty = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    written_data = get_data_frame_to_write(data_empty, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_vadere,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            15.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
    ],
)
def test_load_trajectory_from_vadere_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    expected_data = pd.DataFrame(
        data=data,
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        frame_rate=expected_frame_rate,
        data=written_data,
    )
    expected_data = prepare_data_frame(expected_data)

    traj_data_from_file = load_trajectory_from_vadere(
        trajectory_file=trajectory_vadere,
        frame_rate=expected_frame_rate,
    )

    assert (traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy() == expected_data.to_numpy()).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_vadere_columns_missing(
    tmp_path: pathlib.Path,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "FOO!"],
    )

    written_data = get_data_frame_to_write(data_with_missing_column, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(trajectory_file=trajectory_vadere, frame_rate=24.0)
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_vadere_columns_non_unique(
    tmp_path: pathlib.Path,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 5, 1], [0, 1, -5, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "startX-PID2", Y_COL],
    )

    written_data = get_data_frame_to_write(data_with_missing_column, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(
            trajectory_file=trajectory_vadere,
            frame_rate=24.0,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_vadere_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(trajectory_file=pathlib.Path("non_existing_file"), frame_rate=10)
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_vadere_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(trajectory_file=tmp_path, frame_rate=10)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "area_poly, margin, bounding_box",
    [
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            0,
            0,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            1e-6,
            0,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            0,
            0.5,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            1e-6,
            0.5,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            0,
            0,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            1e-6,
            0,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            0,
            0.5,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 0), (1, 1), (0, 1)]],  # rectangular hole
            ),
            1e-6,
            0.5,
        ),
    ],
)
def test_load_walkable_area_from_vadere_scenario_success(
    tmp_path: pathlib.Path,
    area_poly: shapely.Polygon,
    margin: bool,
    bounding_box: float,
):
    file_path = pathlib.Path(tmp_path / "vadere_test.scenario")
    decimals = 6

    complete_area = area_poly.exterior
    if len(area_poly.interiors) > 0:
        obstacles = area_poly.interiors
    else:
        obstacles = []

    write_vadere_scenario_file(
        file=file_path,
        complete_area=complete_area,
        obstacles=obstacles,
        bounding_box_width=bounding_box,
    )
    walkable_area_from_file = load_walkable_area_from_vadere_scenario(
        file_path,
        margin=margin,
        decimals=decimals,
    )

    # convert test input to expected WalkableArea
    # 1) shrink expected walkable area to area without bounding box
    expected_shell = shapely.Polygon(area_poly.exterior)
    expected_shell = expected_shell.buffer(
        distance=-bounding_box + margin,
        single_sided=True,
        join_style="mitre",
    )
    # convert polygon shell to points to use it as input to WalkableArea
    expected_shell_points = list(expected_shell.exterior.coords)
    # handle floating point errors
    expected_shell_points = np.round(expected_shell_points, decimals)
    # 2) treat holes separately to avoid buffering of holes
    expected_holes = [list(p.coords) for p in area_poly.interiors]
    expected_walkable_area = WalkableArea(expected_shell_points, obstacles=expected_holes)

    expected_walkable_area_polygon = force_cw(expected_walkable_area.polygon)
    walkable_area_from_file_polygon = force_cw(walkable_area_from_file.polygon)

    # The polygons coordinates may differ by up to 'margin' on both axes.
    # The maximum possible difference between coordinates is therefore the diagonal of the square formed
    maximum_coordinate_difference_due_to_margin = math.sqrt(margin * margin + margin * margin)

    assert expected_walkable_area_polygon.equals_exact(
        walkable_area_from_file_polygon,
        maximum_coordinate_difference_due_to_margin,
    )


def test_load_walkable_area_from_vadere_scenario_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_vadere_scenario(vadere_scenario_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_walkable_area_from_vadere_scenario_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_vadere_scenario(vadere_scenario_file=tmp_path)

    assert "is not a file" in str(error_info.value)
