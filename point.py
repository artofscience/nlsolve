from __future__ import annotations

from copy import deepcopy

import numpy as np

State = np.ndarray | float


class Point:
    def __init__(self, uf: State = 0.0, up: State = 0.0, ff: State = 0.0, fp: State = 0.0, y: float = 0.0) -> None:
        """
        Initialize an (equilibrium) point given it's load and corresponding motion in partitioned format.

        :param uf: free / unknown motion
        :param up: prescribed motion
        :param ff: external / applied load
        :param fp: reaction load
        :param y: load proportionality parameter
        """
        self.uf = uf
        self.up = up
        self.ff = ff
        self.fp = fp
        self.y = y

    def __iadd__(self, other: Point) -> Point:
        """
        Adds the content of another Point to this Point.

        :param other: another Point object
        :return: sum of Points
        """
        self.uf += other.uf
        self.up += other.up
        self.ff += other.ff
        self.fp += other.fp
        self.y += other.y
        return self

    def __rmul__(self, other: Point) -> Point:
        """
        Multiplications of two point entries.

        Note rmul makes a deepcopy of itself!

        :param other: another Point
        :return: a copy of itself with the entries multiplied by the other Points entries
        """
        out = deepcopy(self)
        out.uf *= other
        out.up *= other
        out.ff *= other
        out.fp *= other
        out.y *= other
        return out

    def __add__(self, other: Point) -> Point:
        """
        Addition of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the addition
        """
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out += other
        return out

    def __sub__(self, other: Point) -> Point:
        """
        Substraction of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the substraction
        """
        out = deepcopy(Point(self.uf, self.up, self.ff, self.fp, self.y))
        out -= other
        return out
