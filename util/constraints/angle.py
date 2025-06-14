from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Angle(Constraint):
    s1: Segment
    s2: Segment
    angle: float

    def __init__(self, s1: Segment, s2: Segment, angle: float) -> None:
        super().__init__()
        self.s1 = s1
        self.s2 = s2
        self.angle = angle
        self.priority = 5

    def equation(self) -> float:
        return np.abs(
            np.dot(self.s1.get_versor(), self.s2.get_versor()) - np.cos(self.angle)
        )

    def solution_step(self, step_size: float) -> None:
        angle = self.angle - self.s1.angle(self.s2)
        angle_step = angle * step_size / 2
        self.s1.rotate_around_center(angle_step)
        self.s2.rotate_around_center(-angle_step)

    def get_related_segments(self) -> list:
        return list([self.s1, self.s2])

    def get_related_vertexes(self) -> list:
        return list([self.s1.v1, self.s1.v2, self.s2.v1, self.s2.v2])
