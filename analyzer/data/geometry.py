import logging
from dataclasses import dataclass
from typing import List

import pygeos

log = logging.getLogger(__name__)


@dataclass
class Geometry:
    """Class holding the geometry information of the analysis

    Attributes:
        walkable_area (pygeos.Geometry): area in which the pedestrian walk,
        they are only considered for the analysis when inside this area.
        obstacles (List[pygeos.Geometry]): areas which are excluded from the
        analysis, pedestrians inside these areas will be ignored.
    """

    walkable_area: pygeos.Geometry
    obstacles: List[pygeos.Geometry]

    def __init__(
        self, walkable_area: pygeos.Geometry, obstacles: pygeos.Geometry = None
    ):
        self.obstacles = []
        self.walkable_area = walkable_area

        if obstacles is None:
            obstacles = []

        for obstacle in obstacles:
            self.add_obstacle(obstacle)

        pygeos.prepare(self.walkable_area)

    def add_obstacle(self, obstacle: pygeos.Geometry):
        """Adds an obstacle to the geometry

        Args:
            obstacle (pygeos.Geometry): area which will be excluded from the
            analysis.
        """
        if pygeos.covered_by(obstacle, self.walkable_area):
            self.walkable_area = pygeos.difference(self.walkable_area, obstacle)
            self.obstacles.append(obstacle)
            pygeos.prepare(self.walkable_area)
        else:
            log.warning(
                f"The obstacle {obstacle} is not inside the walkable area of "
                f"the geometry and thus will be ignored!"
            )
