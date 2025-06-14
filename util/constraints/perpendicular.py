from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Perpendicular(Constraint):
    s1: Segment
    s2: Segment

    def __init__(self, s1: Segment, s2: Segment) -> None:
        super().__init__()
        self.s1 = s1
        self.s2 = s2
        self.priority = 4

    def equation(self) -> float:
        return np.dot(self.s1.get_versor(), self.s2.get_versor())

    def solution_step(self, step_size: float) -> None:
        angle = self.s1.angle(self.s2)
        angle_step = -np.sign(angle) * (np.pi / 2 - angle) * step_size / 2
        self.s1.rotate_around_center(angle_step)
        self.s2.rotate_around_center(-angle_step)

    def get_related_segments(self) -> list:
        return list([self.s1, self.s2])

    def get_related_vertexes(self) -> list:
        return list([self.s1.v1, self.s1.v2, self.s2.v1, self.s2.v2])
