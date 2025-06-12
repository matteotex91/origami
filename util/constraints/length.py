from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Length(Constraint):
    se: Segment
    le: float

    def __init__(self, se: Segment) -> None:
        super().__init__()
        self.se = se

    def equation(self) -> float:
        return np.abs(
            np.linalg.norm(self.se.v1.position - self.se.v2.position) - self.le
        )
