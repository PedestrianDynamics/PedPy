"""Description of the used columns names in PedPy."""

from typing import Final

# TrajectoryData
ID_COL: Final = "id"
FRAME_COL: Final = "frame"
X_COL: Final = "x"
Y_COL: Final = "y"
POINT_COL: Final = "point"

DENSITY_COL: Final = "density"
POLYGON_COL: Final = "polygon"
INTERSECTION_COL: Final = "intersection"
COUNT_COL: Final = "num_peds"

CUMULATED_COL: Final = "cumulative_pedestrians"
TIME_COL: Final = "time"
SPEED_COL: Final = "speed"
V_X_COL: Final = "v_x"
V_Y_COL: Final = "v_y"
FLOW_COL: Final = "flow"
MEAN_SPEED_COL: Final = "mean_speed"

FIRST_FRAME_COL: Final = "entering_frame"
LAST_FRAME_COL: Final = "leaving_frame"
NEIGHBORS_COL: Final = "neighbors"
DISTANCE_COL: Final = "distance"
CROSSING_FRAME_COL: Final = "crossing_frame"
START_POSITION_COL: Final = "start_position"
END_POSITION_COL: Final = "end_position"
WINDOW_SIZE_COL: Final = "window_size"
