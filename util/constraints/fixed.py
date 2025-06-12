from util.constraints.constraint import Constraint
from util.vertex import Vertex
import numpy as np


class Fixed(Constraint):
    v: Vertex
    pos: np.ndarray

    def __init__(self) -> None:
        super().__init__()
