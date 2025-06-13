from util.constraints.constraint import Constraint
from util.segment import Segment
from util.vertex import Vertex
import numpy as np


class Angle(Constraint):
    s1: Segment
    s2: Segment
    angle: float

    def __init__(self, s1: Segment, s2: Segment) -> None:
        super().__init__()
        self.priority = 5

    def equation(self) -> float:
        return np.abs(
            np.dot(self.s1.get_versor(), self.s2.get_versor()) - np.cos(self.angle)
        )
