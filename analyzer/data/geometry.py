import logging
from dataclasses import dataclass
from typing import List

import pygeos

log = logging.getLogger(__name__)

@dataclass
class Geometry:
    walkable_area: pygeos.Geometry
    obstacles: List[pygeos.Geometry]

    def __init__(self, walkable_area: pygeos.Geometry, obstacles: List[pygeos.Geometry] = []):
        self.walkable_area = walkable_area
        self.obstacles = obstacles

    def add_obstacle(self, obstacle: pygeos.Geometry):
        if pygeos.covered_by(obstacle, self.walkable_area):

            self.walkable_area = pygeos.difference(self.walkable_area, obstacle)
            self.obstacles.append(obstacle)
        else:
            log.warning("Not covered!")
