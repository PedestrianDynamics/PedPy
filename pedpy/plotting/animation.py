"""Module containing Jupyter notebook animation functions for trajectories and Voronoi cells."""

import importlib
from typing import Any, Final, Optional, Tuple

import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

from pedpy.column_identifier import (
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
    INTERSECTION_COL,
    POLYGON_COL,
    SPEED_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import MeasurementArea, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import PedPyRuntimeError, PedPyTypeError, PedPyValueError
from pedpy.plotting.plotting import (
    PEDPY_BLUE,
    PEDPY_RED,
    _plot_polygon,
    plot_measurement_setup,
    plot_walkable_area,
)

_DEFAULT_FPS: Final = 10.0
_DEFAULT_MARKER_SIZE: Final = 30.0
_DEFAULT_VORONOI_PED_SIZE: Final = 10.0
_DEFAULT_LINE_WIDTH: Final = 1.0
_DEFAULT_ALPHA: Final = 1.0
_BOUNDS_MARGIN_FRACTION: Final = 0.05
_MIN_SPAN: Final = 1.0
_MILLISECONDS_PER_SECOND: Final = 1000.0
_COLORBAR_SHRINK: Final = 0.8


def _require_notebook_html_renderer() -> Any:
    """Validate notebook HTML availability and return ``IPython.display.HTML``."""
    try:
        ipython_module = importlib.import_module("IPython")
        ipython_display = importlib.import_module("IPython.display")
    except ImportError as exc:
        raise PedPyRuntimeError(
            "Animating in Jupyter notebooks requires 'ipython'. "
            "Install it with `pip install pedpy[notebook]` (or `pip install ipython`), "
            "or pass `as_html=False` to obtain the `matplotlib.animation.FuncAnimation` object."
        ) from exc

    if ipython_module.get_ipython() is None:
        raise PedPyRuntimeError(
            "HTML animation rendering requires an active IPython/Jupyter notebook environment. "
            "Run inside a Jupyter notebook or pass `as_html=False` to obtain the "
            "`matplotlib.animation.FuncAnimation` object."
        )

    return ipython_display.HTML


def _render_html(anim: FuncAnimation, fig: matplotlib.figure.Figure, html_cls: Any) -> Any:
    """Render ``FuncAnimation`` as an ``IPython.display.HTML`` widget."""
    html_output = html_cls(anim.to_jshtml())
    plt.close(fig)
    return html_output


def _select_frames(df: pd.DataFrame, every_nth_frame: int) -> np.ndarray:
    """Validate ``every_nth_frame`` and return sampled sorted frame numbers."""
    if every_nth_frame < 1:
        raise PedPyValueError(f"'every_nth_frame' must be >= 1, got {every_nth_frame}.")
    if df.empty:
        raise PedPyValueError("Cannot create an animation from empty data.")
    unique_frames = np.sort(df[FRAME_COL].unique())
    return unique_frames[::every_nth_frame]


def _resolve_color_dataframe(
    base_df: pd.DataFrame,
    color_data: Optional[pd.DataFrame],
    color_column: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Merge optional ``color_data`` and validate ``color_column``."""
    resolved_col = color_column
    merged = base_df

    if color_data is not None:
        missing_keys = {ID_COL, FRAME_COL} - set(color_data.columns)
        if missing_keys:
            raise PedPyValueError(
                f"'color_data' must contain '{ID_COL}' and '{FRAME_COL}' columns, missing: {sorted(missing_keys)}."
            )
        candidate_cols = [col for col in color_data.columns if col not in (ID_COL, FRAME_COL)]
        if resolved_col is None:
            if len(candidate_cols) == 1:
                resolved_col = str(candidate_cols[0])
            else:
                raise PedPyValueError("Please specify 'color_column' when 'color_data' has multiple value columns.")
        if resolved_col not in color_data.columns:
            raise PedPyValueError(f"Column '{resolved_col}' not found in 'color_data'.")

        merge_cols = [ID_COL, FRAME_COL, resolved_col]
        merged = base_df.drop(columns=[resolved_col], errors="ignore").merge(
            color_data[merge_cols],
            on=[ID_COL, FRAME_COL],
            how="left",
        )
    elif resolved_col is not None and resolved_col not in base_df.columns:
        raise PedPyValueError(f"Column '{resolved_col}' not found in input data.")

    return merged, resolved_col


def _default_colorbar_label(color_column: str, custom_label: Optional[str]) -> str:
    """Choose a sensible default colorbar label for ``color_column``."""
    if custom_label is not None:
        return custom_label
    if color_column == SPEED_COL:
        return "v / m/s"
    if color_column == DENSITY_COL:
        return r"$\rho$ / $1/m^2$"
    return color_column


def _build_scalar_mappable(
    series: pd.Series,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> mpl.cm.ScalarMappable:
    """Construct a ``ScalarMappable`` from finite bounds of ``series``."""
    finite_vals = series.dropna()
    resolved_vmin = float(finite_vals.min()) if vmin is None and not finite_vals.empty else (vmin or 0.0)
    resolved_vmax = float(finite_vals.max()) if vmax is None and not finite_vals.empty else (vmax or _MIN_SPAN)
    norm = mpl.colors.Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    mappable.set_array([])
    return mappable


def _set_xy_bounds_from_points(axes: matplotlib.axes.Axes, df: pd.DataFrame) -> None:
    """Configure static ``(xlim, ylim)`` bounds from trajectory coordinates."""
    x_min, x_max = float(df[X_COL].min()), float(df[X_COL].max())
    y_min, y_max = float(df[Y_COL].min()), float(df[Y_COL].max())
    span_x = max(x_max - x_min, _MIN_SPAN)
    span_y = max(y_max - y_min, _MIN_SPAN)
    axes.set_xlim(x_min - _BOUNDS_MARGIN_FRACTION * span_x, x_max + _BOUNDS_MARGIN_FRACTION * span_x)
    axes.set_ylim(y_min - _BOUNDS_MARGIN_FRACTION * span_y, y_max + _BOUNDS_MARGIN_FRACTION * span_y)


def _setup_trajectory_figure(
    df: pd.DataFrame,
    walkable_area: Optional[WalkableArea],
    color_column: Optional[str],
    cmap: str,
    kwargs: dict[str, Any],
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Optional[mpl.cm.ScalarMappable], str]:
    """Create figure, static background, and optional colorbar for trajectory animation."""
    figsize = kwargs.pop("figsize", (8, 6))
    base_title = str(kwargs.pop("title", ""))
    x_label = kwargs.pop("x_label", r"x / m")
    y_label = kwargs.pop("y_label", r"y / m")
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    show_colorbar = kwargs.pop("show_colorbar", True)
    cb_label = kwargs.pop("cb_label", None)

    fig, axes = plt.subplots(figsize=figsize)
    if walkable_area is not None:
        plot_walkable_area(walkable_area=walkable_area, axes=axes, **kwargs)
    else:
        _set_xy_bounds_from_points(axes, df)

    axes.set_aspect("equal")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    mappable: Optional[mpl.cm.ScalarMappable] = None
    if color_column is not None:
        mappable = _build_scalar_mappable(df[color_column], cmap=cmap, vmin=vmin, vmax=vmax)
        if show_colorbar:
            fig.colorbar(
                mappable,
                ax=axes,
                label=_default_colorbar_label(color_column, cb_label),
                shrink=_COLORBAR_SHRINK,
            )

    return fig, axes, mappable, base_title


def animate_trajectories(
    *,
    traj: TrajectoryData,
    walkable_area: Optional[WalkableArea] = None,
    color_data: Optional[pd.DataFrame] = None,
    color_column: Optional[str] = None,
    cmap: str = "viridis",
    every_nth_frame: int = 1,
    fps: Optional[float] = None,
    as_html: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a frame-by-frame animation of pedestrian trajectories.

    In Jupyter notebooks (`as_html=True`), returns an interactive
    :class:`IPython.display.HTML` widget built with
    :class:`matplotlib.animation.FuncAnimation`. Pass ``as_html=False`` to
    obtain the underlying :class:`~matplotlib.animation.FuncAnimation` object
    outside a notebook environment.

    Args:
        traj (TrajectoryData): Trajectory object to animate.
        walkable_area (WalkableArea, optional): Walkable area drawn as static
            background.
        color_data (pd.DataFrame, optional): Per-pedestrian, per-frame
            DataFrame containing ``id``, ``frame``, and a scalar column (e.g.
            from :func:`~pedpy.methods.speed_calculator.compute_individual_speed`)
            used to color pedestrian markers.
        color_column (str, optional): Column name in ``color_data`` (or in
            ``traj.data`` when ``color_data`` is ``None``) used for marker
            colors. If ``color_data`` has a single value column besides ``id``
            and ``frame``, it is inferred automatically.
        cmap (str): Matplotlib colormap name used when ``color_column`` is
            active (default: ``"viridis"``).
        every_nth_frame (int): Step through sorted unique frames (default: 1).
        fps (float, optional): Playback speed in frames per second (defaults to
            ``traj.frame_rate / every_nth_frame``).
        as_html (bool): Render as :class:`IPython.display.HTML` for Jupyter
            notebooks when ``True`` (default), or return the
            :class:`~matplotlib.animation.FuncAnimation` instance when
            ``False``.
        **kwargs: Optional styling parameters aligned with
            :func:`~pedpy.plotting.plotting.plot_trajectories`
            (``marker_size`` / ``traj_frame_marker_size``, ``marker`` /
            ``traj_frame_marker``, ``traj_color``, ``traj_alpha``, ``vmin``,
            ``vmax``, ``show_colorbar``, ``cb_label``, ``figsize``, ``title``,
            ``x_label``, ``y_label``, ``border_line_color``,
            ``border_line_width``, ``hole_color``, ``hole_alpha``).

    Returns:
        ``IPython.display.HTML`` when ``as_html=True``, otherwise
        ``matplotlib.animation.FuncAnimation``.
    """
    if not isinstance(traj, TrajectoryData):
        raise PedPyTypeError(f"Expected 'traj' to be a TrajectoryData, got {type(traj).__name__!r} instead.")

    html_cls = _require_notebook_html_renderer() if as_html else None
    requested_col = color_column if color_column is not None else kwargs.pop("color_by_column", None)
    frames = _select_frames(traj.data, every_nth_frame)
    plot_df, resolved_col = _resolve_color_dataframe(traj.data, color_data, requested_col)

    marker_size = kwargs.pop("marker_size", kwargs.pop("traj_frame_marker_size", _DEFAULT_MARKER_SIZE))
    marker = kwargs.pop("marker", kwargs.pop("traj_frame_marker", "o"))
    traj_color = kwargs.pop("traj_color", PEDPY_RED)
    traj_alpha = kwargs.pop("traj_alpha", _DEFAULT_ALPHA)

    fig, axes, mappable, base_title = _setup_trajectory_figure(
        df=plot_df,
        walkable_area=walkable_area,
        color_column=resolved_col,
        cmap=cmap,
        kwargs=kwargs,
    )

    scatter = axes.scatter(
        [],
        [],
        s=marker_size,
        marker=marker,
        color=traj_color,
        alpha=traj_alpha,
        zorder=3,
    )
    grouped = {int(frame_val): group for frame_val, group in plot_df.groupby(FRAME_COL)}

    def _update(frame_val: int) -> Tuple[Artist, ...]:
        frame_rows = grouped.get(int(frame_val))
        if frame_rows is None or frame_rows.empty:
            scatter.set_offsets(np.empty((0, 2)))
        else:
            coords = frame_rows[[X_COL, Y_COL]].to_numpy(dtype=float)
            scatter.set_offsets(coords)
            if mappable is not None and resolved_col is not None:
                rgba = mappable.to_rgba(frame_rows[resolved_col].to_numpy(dtype=float))
                rgba_list = [(float(r), float(g), float(b), float(a)) for r, g, b, a in rgba]
                scatter.set_facecolor(rgba_list)
                scatter.set_edgecolor(rgba_list)
        title_prefix = f"{base_title} - " if base_title else ""
        axes.set_title(f"{title_prefix}frame {int(frame_val)}")
        return (scatter,)

    effective_fps = fps if fps is not None else max(traj.frame_rate / every_nth_frame, _MIN_SPAN)
    interval_ms = _MILLISECONDS_PER_SECOND / effective_fps
    anim = FuncAnimation(fig, _update, frames=frames, interval=interval_ms, blit=False)

    if as_html:
        return _render_html(anim, fig, html_cls)
    return anim


def _setup_voronoi_figure(
    voronoi_df: pd.DataFrame,
    walkable_area: Optional[WalkableArea],
    measurement_area: Optional[MeasurementArea],
    color_column: Optional[str],
    cmap: str,
    kwargs: dict[str, Any],
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Optional[mpl.cm.ScalarMappable], str]:
    """Create figure, static geometry, and optional colorbar for Voronoi animation."""
    figsize = kwargs.pop("figsize", (8, 6))
    base_title = str(kwargs.pop("title", ""))
    x_label = kwargs.pop("x_label", r"x / m")
    y_label = kwargs.pop("y_label", r"y / m")
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    show_colorbar = kwargs.pop("show_colorbar", True)
    cb_label = kwargs.pop("cb_label", None)

    fig, axes = plt.subplots(figsize=figsize)
    if measurement_area is not None:
        plot_measurement_setup(measurement_areas=[measurement_area], axes=axes, **kwargs)

    if walkable_area is not None:
        plot_walkable_area(walkable_area=walkable_area, axes=axes, **kwargs)
    else:
        bounds = shapely.total_bounds(voronoi_df[POLYGON_COL].to_numpy())
        span_x = max(float(bounds[2] - bounds[0]), _MIN_SPAN)
        span_y = max(float(bounds[3] - bounds[1]), _MIN_SPAN)
        axes.set_xlim(
            float(bounds[0]) - _BOUNDS_MARGIN_FRACTION * span_x, float(bounds[2]) + _BOUNDS_MARGIN_FRACTION * span_x
        )
        axes.set_ylim(
            float(bounds[1]) - _BOUNDS_MARGIN_FRACTION * span_y, float(bounds[3]) + _BOUNDS_MARGIN_FRACTION * span_y
        )

    axes.set_aspect("equal")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    mappable: Optional[mpl.cm.ScalarMappable] = None
    if color_column is not None:
        mappable = _build_scalar_mappable(voronoi_df[color_column], cmap=cmap, vmin=vmin, vmax=vmax)
        if show_colorbar:
            fig.colorbar(
                mappable,
                ax=axes,
                label=_default_colorbar_label(color_column, cb_label),
                shrink=_COLORBAR_SHRINK,
            )

    return fig, axes, mappable, base_title


def _draw_voronoi_frame(
    axes: matplotlib.axes.Axes,
    frame_rows: pd.DataFrame,
    mappable: Optional[mpl.cm.ScalarMappable],
    color_column: Optional[str],
    style: dict[str, Any],
) -> None:
    """Render Voronoi polygons for a single animation frame onto ``axes``."""
    for _, row in frame_rows.iterrows():
        poly = row[POLYGON_COL]
        if mappable is not None and color_column is not None:
            fill_color: Any = tuple(mappable.to_rgba(np.asarray([row[color_column]], dtype=float))[0])
        else:
            fill_color = "white"

        _plot_polygon(
            axes=axes,
            polygon=poly,
            line_color=style["border_color"],
            line_width=style["border_width"],
            polygon_color=fill_color,
            polygon_alpha=style["outside_alpha"],
            zorder=1,
        )
        if INTERSECTION_COL in frame_rows.columns and not shapely.is_empty(row[INTERSECTION_COL]):
            _plot_polygon(
                axes=axes,
                polygon=row[INTERSECTION_COL],
                line_color="none",
                polygon_color=fill_color,
                polygon_alpha=style["inside_alpha"],
                zorder=1,
            )


def animate_voronoi_cells(
    *,
    voronoi_data: pd.DataFrame,
    traj: Optional[TrajectoryData] = None,
    walkable_area: Optional[WalkableArea] = None,
    measurement_area: Optional[MeasurementArea] = None,
    color_data: Optional[pd.DataFrame] = None,
    color_column: Optional[str] = None,
    cmap: str = "YlGn",
    every_nth_frame: int = 1,
    as_html: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a frame-by-frame animation of individual Voronoi cells.

    In Jupyter notebooks (`as_html=True`), returns an interactive
    :class:`IPython.display.HTML` widget built with
    :class:`matplotlib.animation.FuncAnimation`. Pass ``as_html=False`` to
    obtain the underlying :class:`~matplotlib.animation.FuncAnimation` object
    outside a notebook environment.

    Args:
        voronoi_data (pd.DataFrame): Voronoi polygon DataFrame containing
            ``id``, ``frame``, and ``polygon`` columns.
        traj (TrajectoryData, optional): Trajectory data used to overlay
            pedestrian positions and infer playback frame rate (also accepted
            as ``traj_data`` in ``kwargs``).
        walkable_area (WalkableArea, optional): Walkable area drawn as static
            background.
        measurement_area (MeasurementArea, optional): Measurement area drawn as
            static background.
        color_data (pd.DataFrame, optional): Per-pedestrian, per-frame
            DataFrame merged onto ``voronoi_data`` to color Voronoi cells.
        color_column (str, optional): Column name in ``color_data`` or
            ``voronoi_data`` used to color cells (also accepted as
            ``color_by_column`` in ``kwargs``).
        cmap (str): Colormap name used when ``color_column`` is active
            (default: ``"YlGn"``).
        every_nth_frame (int): Step through sorted unique frames (default: 1).
        as_html (bool): Render as :class:`IPython.display.HTML` for Jupyter
            notebooks when ``True`` (default), or return the
            :class:`~matplotlib.animation.FuncAnimation` instance when
            ``False``.
        **kwargs: Optional styling parameters aligned with
            :func:`~pedpy.plotting.plotting.plot_voronoi_cells` (``fps``,
            ``ped_color``, ``ped_size``, ``voronoi_border_color``,
            ``voronoi_border_width``, ``voronoi_inside_ma_alpha``,
            ``voronoi_outside_ma_alpha``, ``vmin``, ``vmax``,
            ``show_colorbar``, ``cb_label``, ``figsize``, ``title``,
            ``x_label``, ``y_label``).

    Returns:
        ``IPython.display.HTML`` when ``as_html=True``, otherwise
        ``matplotlib.animation.FuncAnimation``.
    """
    html_cls = _require_notebook_html_renderer() if as_html else None
    resolved_traj: Optional[TrajectoryData] = traj if traj is not None else kwargs.pop("traj_data", None)
    requested_col = color_column if color_column is not None else kwargs.pop("color_by_column", None)
    fps: Optional[float] = kwargs.pop("fps", None)

    style = {
        "ped_color": kwargs.pop("ped_color", PEDPY_BLUE),
        "ped_size": kwargs.pop("ped_size", _DEFAULT_VORONOI_PED_SIZE),
        "border_color": kwargs.pop("voronoi_border_color", PEDPY_BLUE),
        "border_width": kwargs.pop("voronoi_border_width", _DEFAULT_LINE_WIDTH),
        "inside_alpha": kwargs.pop("voronoi_inside_ma_alpha", _DEFAULT_ALPHA),
        "outside_alpha": kwargs.pop("voronoi_outside_ma_alpha", _DEFAULT_ALPHA),
    }

    frames = _select_frames(voronoi_data, every_nth_frame)
    plot_df, resolved_col = _resolve_color_dataframe(voronoi_data, color_data, requested_col)
    if resolved_traj is not None:
        plot_df = plot_df.drop(columns=[X_COL, Y_COL], errors="ignore").merge(
            resolved_traj.data[[ID_COL, FRAME_COL, X_COL, Y_COL]],
            on=[ID_COL, FRAME_COL],
            how="left",
        )

    fig, axes, mappable, base_title = _setup_voronoi_figure(
        voronoi_df=plot_df,
        walkable_area=walkable_area,
        measurement_area=measurement_area,
        color_column=resolved_col,
        cmap=cmap,
        kwargs=kwargs,
    )

    static_patch_count = len(axes.patches)
    static_line_count = len(axes.lines)
    scatter = axes.scatter([], [], color=style["ped_color"], s=style["ped_size"], zorder=3)
    grouped = {int(frame_val): group for frame_val, group in plot_df.groupby(FRAME_COL)}

    def _update(frame_val: int) -> Tuple[Artist, ...]:
        for patch in list(axes.patches[static_patch_count:]):
            patch.remove()
        for line in list(axes.lines[static_line_count:]):
            line.remove()

        frame_rows = grouped.get(int(frame_val))
        if frame_rows is None or frame_rows.empty:
            scatter.set_offsets(np.empty((0, 2)))
        else:
            _draw_voronoi_frame(axes, frame_rows, mappable, resolved_col, style)
            if resolved_traj is not None and X_COL in frame_rows.columns:
                scatter.set_offsets(frame_rows[[X_COL, Y_COL]].to_numpy(dtype=float))

        title_prefix = f"{base_title} - " if base_title else ""
        axes.set_title(f"{title_prefix}frame {int(frame_val)}")
        return (scatter,)

    default_fps = resolved_traj.frame_rate / every_nth_frame if resolved_traj is not None else _DEFAULT_FPS
    effective_fps = fps if fps is not None else max(default_fps, _MIN_SPAN)
    anim = FuncAnimation(fig, _update, frames=frames, interval=_MILLISECONDS_PER_SECOND / effective_fps, blit=False)

    if as_html:
        return _render_html(anim, fig, html_cls)
    return anim
