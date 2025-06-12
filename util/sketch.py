import numpy as np
from util.vertex import Vertex
from util.segment import Segment


class Sketch:
    vertices: list
    segments: list

    def __init__(self) -> None:
        self.vertices = list()
        self.segments = list()

    def add_vertex(self, position: np.ndarray):
        v = Vertex(position)
        self.vertices.append(v)

    def add_segment(self, v1: Vertex, v2: Vertex):
        if v1 in self.vertices and v2 in self.vertices:
            s = Segment(v1, v2)
            self.segments.append(s)

    def get_closer_vertex(self, position: np.ndarray):
        if len(self.vertices) == 0:
            return None
        elif len(self.vertices) == 1:
            return self.vertices[0]
        else:
            distances = [np.linalg.norm(position - v.position) for v in self.vertices]
            return self.vertices[np.argmin(distances)]
