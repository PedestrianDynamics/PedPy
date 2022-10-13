import logging
from dataclasses import dataclass
from typing import List

import shapely
from shapely import Polygon

log = logging.getLogger(__name__)


@dataclass
class Geometry:
    """Class holding the geometry information of the analysis

    Attributes:
        walkable_area (shapely.Polygon): area in which the pedestrian walk,
        they are only considered for the analysis when inside this area.
        obstacles (List[shapely.Polygon]): areas which are excluded from the
        analysis, pedestrians inside these areas will be ignored.
    """

    walkable_area: Polygon
    obstacles: List[Polygon]

    def __init__(
        self, *, walkable_area: Polygon, obstacles: List[Polygon] = None
    ):
        self.obstacles = []
        self.walkable_area = walkable_area

        if obstacles is None:
            obstacles = []

        for obstacle in obstacles:
            self.add_obstacle(obstacle)

        shapely.prepare(self.walkable_area)

    def add_obstacle(self, obstacle: Polygon):
        """Adds an obstacle to the geometry

        Args:
            obstacle (Polygon): area which will be excluded from the
            analysis.
        """
        if obstacle.within(self.walkable_area):
            self.walkable_area = shapely.difference(
                self.walkable_area, obstacle
            )
            self.obstacles.append(obstacle)
            shapely.prepare(self.walkable_area)
        else:
            log.warning(
                f"The obstacle {obstacle} is not inside the walkable area of "
                f"the geometry and thus will be ignored!"
            )
