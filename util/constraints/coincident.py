from util.constraints.constraint import Constraint
from util.vertex import Vertex
import numpy as np


class Coincident(Constraint):
    v1: Vertex
    v2: Vertex

    def __init__(self, v1: Vertex, v2: Vertex) -> None:
        super().__init__()
        self.priority = 1

    def equation(self) -> float:
        return float(np.linalg.norm(self.v1.position - self.v2.position))

    def solution_step(self, step_size: float) -> None:
        delta = step_size * (self.v2.position - self.v1.position) / 2
        self.v1.position = self.v1.position + delta
        self.v2.position = self.v2.position - delta
