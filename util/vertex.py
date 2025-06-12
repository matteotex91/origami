import numpy as np


class Vertex:
    position: np.ndarray

    def __init__(self, position: np.ndarray) -> None:
        self.position = position
