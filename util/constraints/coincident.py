from util.constraints.constraint import Constraint
from util.vertex import Vertex
import numpy as np


class Coincident(Constraint):
    v1: Vertex
    v2: Vertex

    def __init__(self) -> None:
        super().__init__()

    def equation(self) -> float:
        return float(np.linalg.norm(self.v1.position - self.v2.position))
