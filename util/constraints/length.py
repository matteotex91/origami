from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Length(Constraint):
    se: Segment
    le: float

    def __init__(self, se: Segment, le: float) -> None:
        super().__init__()
        self.se = se
        self.le = le
        self.priority = 2

    def equation(self) -> float:
        return np.abs(
            np.linalg.norm(self.se.v1.position - self.se.v2.position) - self.le
        )

    def solution_step(self, step_size: float) -> None:
        delta = self.se.v2.position - self.se.v1.position
        delta = delta * (1 - self.le / np.linalg.norm(delta)) * step_size / 2
        self.se.v1.position = self.se.v1.position + delta
        self.se.v2.position = self.se.v2.position - delta

    def get_related_segments(self) -> list:
        return list([self.se])

    def get_related_vertexes(self) -> list:
        return list([self.se.v1, self.se.v2])
