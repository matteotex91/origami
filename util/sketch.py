import numpy as np
from util.vertex import Vertex
from util.segment import Segment
from util.constraints.constraint import Constraint
from util.constraints.fixed import Fixed
from util.constraints.perpendicular import Perpendicular
from util.constraints.length import Length


class Sketch:
    vertices: list
    segments: list
    constraints: list

    def __init__(self) -> None:
        self.vertices = list()
        self.segments = list()
        self.constraints = list()

    def add_vertex(self, v: Vertex):
        self.vertices.append(v)

    def add_segment(self, s: Segment):
        self.segments.append(s)

    def add_constraint(self, c: Constraint):
        self.constraints.append(c)

    def get_closer_vertex(self, position: np.ndarray):
        if len(self.vertices) == 0:
            return None
        elif len(self.vertices) == 1:
            return self.vertices[0]
        else:
            distances = [np.linalg.norm(position - v.position) for v in self.vertices]
            return self.vertices[np.argmin(distances)]

    def get_constraints_from_vertex(self, v: Vertex):
        related = []
        for c in self.constraints:
            if v in c.get_related_vertexes() and c not in related:
                related.append(c)
        return related

    def get_constraints_from_segment(self, s: Segment):
        related = []
        for c in self.constraints:
            if s in c.get_related_segments() and c not in related:
                related.append(c)
        return related

    def solve_constraints(
        self, tol: float, max_iter: int = 5000, step_size: float = 1e-2
    ) -> bool:
        if np.max([np.abs(c.equation()) for c in self.constraints]) < tol:
            return True
        priority_map = [(c.priority, c) for c in self.constraints]
        sorted_constraints = [c for (_, c) in priority_map]

        for _ in range(max_iter):
            for c in sorted_constraints:
                c.solution_step(step_size)
            print([np.abs(c.equation()) for c in self.constraints])
            if np.max([np.abs(c.equation()) for c in self.constraints]) < tol:
                return True
        return False


if __name__ == "__main__":
    sk = Sketch()
    v1 = Vertex(np.array([0, 0]))
    v2 = Vertex(np.array([1, 0]))
    v3 = Vertex(np.array([1, 1]))
    s1 = Segment(v1, v2)
    s2 = Segment(v1, v3)
    f1 = Fixed(v1, np.array([0, 0]))
    f2 = Fixed(v2, np.array([1, 0]))
    p = Perpendicular(s1, s2)
    le = Length(s2, 10)
    sk.add_segment(s1)
    sk.add_segment(s2)
    sk.add_constraint(f1)
    sk.add_constraint(f2)
    sk.add_constraint(p)
    sk.add_constraint(le)
    print(sk.solve_constraints(1e-3))
    print(v1.position)
    print(v2.position)
    print(v3.position)
    print(sk.get_constraints_from_segment(s1))
