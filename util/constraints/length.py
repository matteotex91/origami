from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Length(Constraint):
    s: Segment
    l: float

    def __init__(self, s: Segment) -> None:
        super().__init__()
        self.s = s

    def equation(self) -> float:
        return np.abs(np.linalg.norm(self.s.v1.position - self.s.v2.position) - self.l)
