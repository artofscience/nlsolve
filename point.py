from __future__ import annotations
import numpy as np
from copy import deepcopy

State = np.ndarray | float

class Point:
    def __init__(self, uf: State = 0.0, up: State = 0.0, ff: State = 0.0, fp: State = 0.0, y: float = 0.0) -> None:
        self.uf = uf
        self.up = up
        self.ff = ff
        self.fp = fp
        self.y = y

    def __iadd__(self, other: Point) -> Point:
        self.uf += other.uf
        self.up += other.up
        self.ff += other.ff
        self.fp += other.fp
        self.y += other.y
        return self

    def __rmul__(self, other: Point) -> Point:
        out = deepcopy(self)
        out.uf *= other
        out.up *= other
        out.ff *= other
        out.fp *= other
        out.y *= other
        return out

    def __add__(self, other: Point) -> Point:
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out += other
        return out

    def __sub__(self, other: Point) -> Point:
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out -= other
        return out
