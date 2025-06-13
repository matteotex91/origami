from util.constraints.constraint import Constraint
from util.vertex import Vertex
import numpy as np


class Fixed(Constraint):
    v: Vertex
    pos: np.ndarray

    def __init__(self) -> None:
        super().__init__()
        self.priority = 0

    def equation(self) -> float:
        return float(np.linalg.norm(self.v.position - self.pos))

    def solution_step(self, step_size: float) -> None:
        self.v.position = self.pos
