from util.constraints.constraint import Constraint
from util.segment import Segment
import numpy as np


class Parallel(Constraint):
    s1: Segment
    s2: Segment

    def __init__(self) -> None:
        super().__init__()

    def equation(self) -> float:
        return np.dot(self.s1.get_versor(), self.s2.get_versor()) - 1
