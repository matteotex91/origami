from __future__ import annotations
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

    def rotate_around_center(self, angle):
        R = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                ],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        v1 = self.v1.position - self.vm.position
        v2 = self.v2.position - self.vm.position
        v1 = np.dot(R, v1)
        v2 = np.dot(R, v2)
        self.v1.position = self.vm.position + v1
        self.v2.position = self.vm.position + v2

    def angle(self, s: Segment) -> float:
        v1 = self.get_versor()
        v2 = s.get_versor()
        cos = np.dot(v1, v2)
        sin = float(np.cross(v1, v2))
        return np.arctan2(sin, cos)
