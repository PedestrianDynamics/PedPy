import importlib
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation
from tests.utils.utils import get_trajectory

from pedpy import (
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
    SPEED_COL,
    MeasurementArea,
    PedPyRuntimeError,
    PedPyTypeError,
    PedPyValueError,
    SpeedCalculation,
    TrajectoryData,
    WalkableArea,
    animate_trajectories,
    animate_voronoi_cells,
    compute_individual_speed,
    compute_individual_voronoi_polygons,
    compute_voronoi_density,
)


def _sample_setup() -> tuple[TrajectoryData, WalkableArea, MeasurementArea]:
    traj_df = get_trajectory(
        shape=[2, 2],
        number_frames=6,
        start_position=np.array([0.0, 0.0]),
        movement_direction=np.array([0.25, 0.0]),
        ped_distance=1.5,
    )
    traj = TrajectoryData(data=traj_df, frame_rate=10.0)
    walkable_area = WalkableArea([(-2, -2), (8, -2), (8, 5), (-2, 5)])
    measurement_area = MeasurementArea([(0, -1), (4, -1), (4, 3), (0, 3)])
    return traj, walkable_area, measurement_area


def test_animate_trajectories_func_animation_and_color_data():
    traj, walkable_area, _ = _sample_setup()
    speed_df = compute_individual_speed(
        traj_data=traj,
        frame_step=1,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    anim = animate_trajectories(
        traj=traj,
        walkable_area=walkable_area,
        color_data=speed_df,
        every_nth_frame=2,
        marker_size=40.0,
        title="Corridor",
        as_html=False,
    )
    try:
        assert isinstance(anim, FuncAnimation)
        sampled_frames = list(anim.new_frame_seq())
        assert sampled_frames == [0, 2, 4]

        # Render the animation frames via jshtml to exercise _update across all frames
        jshtml = anim.to_jshtml()
        assert "Animation" in jshtml
        assert anim._fig.axes[0].get_title() == "Corridor - frame 4"
    finally:
        plt.close(anim._fig)


def test_animate_trajectories_color_from_trajectory_column():
    traj, _, _ = _sample_setup()
    df_with_extra = traj.data.copy()
    df_with_extra["custom_metric"] = df_with_extra[FRAME_COL].astype(float) * 0.5
    traj_extra = TrajectoryData(data=df_with_extra, frame_rate=traj.frame_rate)

    anim = animate_trajectories(
        traj=traj_extra,
        color_column="custom_metric",
        cmap="plasma",
        as_html=False,
    )
    try:
        assert isinstance(anim, FuncAnimation)
        anim._func(3)
        assert anim._fig.axes[0].get_title() == "frame 3"
    finally:
        plt.close(anim._fig)


def test_animate_voronoi_cells_func_animation():
    traj, walkable_area, measurement_area = _sample_setup()
    voronoi_polys = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area)
    _, intersecting = compute_voronoi_density(
        individual_voronoi_data=voronoi_polys,
        measurement_area=measurement_area,
    )
    speed_df = compute_individual_speed(
        traj_data=traj,
        frame_step=1,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    anim = animate_voronoi_cells(
        voronoi_data=intersecting,
        traj=traj,
        walkable_area=walkable_area,
        measurement_area=measurement_area,
        color_data=speed_df,
        color_column=SPEED_COL,
        every_nth_frame=2,
        as_html=False,
    )
    try:
        assert isinstance(anim, FuncAnimation)
        jshtml = anim.to_jshtml()
        assert "Animation" in jshtml
        assert anim._fig.axes[0].get_title() == "frame 4"
    finally:
        plt.close(anim._fig)


def test_animate_voronoi_cells_without_walkable_area():
    traj, walkable_area, _ = _sample_setup()
    voronoi_polys = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area)

    anim = animate_voronoi_cells(
        voronoi_data=voronoi_polys,
        color_column=DENSITY_COL,
        every_nth_frame=3,
        as_html=False,
    )
    try:
        assert isinstance(anim, FuncAnimation)
        anim._func(3)
        assert anim._fig.axes[0].get_title() == "frame 3"
    finally:
        plt.close(anim._fig)


def test_animate_html_rendering_with_mocked_ipython(monkeypatch: pytest.MonkeyPatch):
    traj, walkable_area, _ = _sample_setup()

    class DummyHTML:
        def __init__(self, data: str) -> None:
            self.data = data

    fake_ipython = types.SimpleNamespace(get_ipython=lambda: object())
    fake_display = types.SimpleNamespace(HTML=DummyHTML)

    real_import_module = importlib.import_module

    def _mock_import_module(name: str, package: str | None = None):
        if name == "IPython":
            return fake_ipython
        if name == "IPython.display":
            return fake_display
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _mock_import_module)

    result = animate_trajectories(
        traj=traj,
        walkable_area=walkable_area,
        every_nth_frame=3,
        as_html=True,
    )
    assert isinstance(result, DummyHTML)
    assert "Animation" in result.data


def test_animate_html_raises_when_ipython_missing(monkeypatch: pytest.MonkeyPatch):
    traj, _, _ = _sample_setup()
    real_import_module = importlib.import_module

    def _missing_ipython(name: str, package: str | None = None):
        if name.startswith("IPython"):
            raise ImportError("No module named 'IPython'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing_ipython)

    with pytest.raises(PedPyRuntimeError, match=r"pedpy\[notebook\]"):
        animate_trajectories(traj=traj, as_html=True)


def test_animate_html_raises_outside_notebook(monkeypatch: pytest.MonkeyPatch):
    traj, _, _ = _sample_setup()
    fake_ipython = types.SimpleNamespace(get_ipython=lambda: None)
    fake_display = types.SimpleNamespace(HTML=lambda data: data)
    real_import_module = importlib.import_module

    def _mock_import_module(name: str, package: str | None = None):
        if name == "IPython":
            return fake_ipython
        if name == "IPython.display":
            return fake_display
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _mock_import_module)

    with pytest.raises(PedPyRuntimeError, match="notebook environment"):
        animate_trajectories(traj=traj, as_html=True)


def test_animate_validation_errors():
    traj, _, _ = _sample_setup()

    with pytest.raises(PedPyTypeError, match="TrajectoryData"):
        animate_trajectories(traj=traj.data, as_html=False)  # type: ignore[arg-type]

    with pytest.raises(PedPyValueError, match="every_nth_frame"):
        animate_trajectories(traj=traj, every_nth_frame=0, as_html=False)

    with pytest.raises(PedPyValueError, match="not found"):
        animate_trajectories(traj=traj, color_column="nonexistent_col", as_html=False)

    multi_col_df = pd.DataFrame({ID_COL: [0], FRAME_COL: [0], "a": [1.0], "b": [2.0]})
    with pytest.raises(PedPyValueError, match="color_column"):
        animate_trajectories(traj=traj, color_data=multi_col_df, as_html=False)
