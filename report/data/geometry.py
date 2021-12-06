from dataclasses import dataclass
from typing import List

from shapely.geometry import LineString


@dataclass
class Geometry:
    walls: List[LineString]
