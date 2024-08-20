from __future__ import annotations
import numpy as np
from copy import deepcopy

State = np.ndarray | float

class Point:
    def __init__(self, u: State = 0.0, v: State = 0.0, f: State = 0.0, p: State = 0.0, y: float = 0.0) -> None:
        self.u = u
        self.v = v
        self.f = f
        self.p = p
        self.y = y

    def __iadd__(self, other: Point) -> Point:
        self.u += other.u
        self.v += other.v
        self.f += other.f
        self.p += other.p
        self.y += other.y
        return self

    def __rmul__(self, other: Point) -> Point:
        out = deepcopy(self)
        out.u *= other
        out.v *= other
        out.f *= other
        out.p *= other
        out.y *= other
        return out

    def __add__(self, other: Point) -> Point:
        out = deepcopy(Point(self.u, self.v, self.f, self.p, self.y))
        out += other
        return out

    def __sub__(self, other: Point) -> Point:
        out = deepcopy(Point(self.u, self.v, self.f, self.p, self.y))
        out -= other
        return out
