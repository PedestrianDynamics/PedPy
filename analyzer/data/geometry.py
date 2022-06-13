from dataclasses import dataclass

import pygeos


@dataclass
class Geometry:
    walkable_area: pygeos.Geometry
