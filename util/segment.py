from util.vertex import Vertex
import numpy as np


class Segment:
    v1: Vertex
    v2: Vertex
    vm: Vertex

    def __init__(self, v1: Vertex, v2: Vertex) -> None:
        self.v1 = v1
        self.v2 = v2
        self.vm = Vertex(0.5 * (v1.position + v2.position))

    def get_length(self) -> float:
        return float(np.linalg.norm(self.v1.position - self.v2.position))

    def get_versor(self):
        return (self.v2.position - self.v1.position) / self.get_length()
