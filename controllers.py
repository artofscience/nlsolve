"""
This file contains objects that act as a controller for the incremental solver.
These objects control the characteristic length of the constraint function used and modify its value.
Different controllers have different behaviour in terms of what observations are used to modify and how to modify its value.
"""

class Controller:
    """
    This controller holds a constant value equal to its initial value,
     both incr and decr functions do not alter the imposed value.
    """

    def __init__(self, value: float = 0.1) -> None:
        self.value = value

    def increase(self) -> None:
        pass

    def decrease(self) -> None:
        pass


class Adaptive(Controller):
    """
    Controller that changes value based on incr and decr values,
    taking into account preset minimum and maximum values.
    """

    def __init__(self, value: float = 0.1,
                 incr: float = 2.0, decr: float = 0.5,
                 min: float = 0.01, max: float = 1.0) -> None:
        super().__init__(value)
        self.incr = incr
        self.decr = decr
        self.min = min
        self.max = max

    def increase(self) -> None:
        self.value = min(self.incr * self.value, self.max)

    def decrease(self) -> None:
        self.value = max(self.decr * self.value, self.min)
