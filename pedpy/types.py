"""Description of the used columns names in PedPy."""

from typing import Final

# TrajectoryData
ID_COL: Final = "ID"
FRAME_COL: Final = "frame"
X_COL: Final = "X"
Y_COL: Final = "Y"
POINT_COL: Final = "points"


# Density Calculator
DENSITY_COL: Final = "density"
POLYGON_COL: Final = "voronoi_polygon"
INTERSECTION_COL: Final = "voronoi_ma_intersection"
COUNT_COL: Final = "num_peds"

# Flow calculator
CUMULATED_COL: Final = "cumulative_pedestrians"
TIME_COL: Final = "time"
SPEED_COL: Final = "speed"
FLOW_COL: Final = "flow"
MEAN_SPEED_COL: Final = "mean_speed"

# Method utils
FIRST_FRAME_COL: Final = "frame_start"
LAST_FRAME_COL: Final = "frame_end"
NEIGHBORS_COL: Final = "neighbors"
DISTANCE_COL: Final = "distance_to_line"
CROSSING_FRAME_COL: Final = "crossing_frame"
INDIVIDUAL_DENSITY_COL: Final = "individual_density"
START_POSITION_COL: Final = "start"
END_POSITION_COL: Final = "end"
WINDOW_SIZE_COL: Final = "window_size"
