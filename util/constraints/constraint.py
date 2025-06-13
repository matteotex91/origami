class Constraint:
    """
    solution priority
    0->fixed
    1->coincident
    2->length
    3->parallel
    4->perpendicular
    5->angle
    """

    priority: int

    def __init__(self) -> None:
        pass

    def equation(self) -> float:
        raise NotImplementedError()

    def solution_step(self, step_size: float) -> None:
        raise NotImplementedError()
