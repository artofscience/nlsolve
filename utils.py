from __future__ import annotations

from abc import ABC
from copy import deepcopy

import numpy as np
State = np.ndarray[float] | None


class Structure(ABC):
    """
    Interface of a nonlinear function to the nonlinear solver.

    The external / internal / residual load, motion and stiffness matrix are partitioned based on the free and prescribed degrees of freedom.
    Both the free and prescribed degrees of freedom can be of dimension 0, 1 or higher.
    If dim(free) = 0, then dim(prescribed) > 0 and vice versa.
    That is, either external_load OR prescribed_motion OR BOTH are to be provided.
    """
    def __init__(self):
        self.ff = self.ff()
        self.up = self.up()

        # get dimension of free and prescribed degrees of freedom
        self.nf = np.shape(self.ff)[0] if self.ff is not None else None
        self.np = np.shape(self.up)[0] if self.up is not None else None

        # squared norm of load external load and prescribed motion
        self.ff2 = np.dot(self.ff, self.ff) if self.nf is not None else None
        self.up2 = np.dot(self.up, self.up) if self.np is not None else None

    def ff(self) -> State:
        """
        Applied external load.

        :return: None
        """
        return None

    def up(self) -> State:
        """
        Prescribed motion.

        :return: None
        """
        return None

    def internal_load_free(self, p: Point) -> State:
        """
        Internal load associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def internal_load_prescribed(self, p: Point) -> State:
        """
        Internal load associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def rf(self, p: Point) -> State:
        """
        Residual associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the free degrees of freedom
        """

        # free residual is defined as the free internal load PLUS the proportional loading parameter times the applied external load
        return self.internal_load_free(p) - p.y * self.ff

    def rp(self, p: Point) -> State:
        """
        Residual associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the prescribed degrees of freedom
        """

        # prescribed residual is defined as the prescribed internal load PLUS the reaction load
        return self.internal_load_prescribed(p) - p.fp

    def kff(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kfp(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the free-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kpf(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def kpp(self, p: Point) -> State:
        """
        Tangent stiffness matrix / Jacobian associated to the prescribed-prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def ddp(self, p: Point, u: np.ndarray, y: float) -> Point:
        """
        Provides the iterative updated state given some iterative load parameter.

        :param p: current state (p + dp)
        :param u: resultants from solve
        :param y: iterative load parameter
        :return:
        """
        dduf, ddup, ddff, ddfp = 0.0, 0.0, 0.0, 0.0

        if self.nf:
            dduf = u[:, 0] + y * u[:, 1]
            ddff = y * self.ff
        if self.np:
            ddup = y * self.up
            ddfp = self.rp(p) + y * self.kpp(p) @ self.up
            ddfp += self.kpf(p) @ dduf if self.nf else 0.0

        return Point(dduf, ddup, ddff, ddfp, y)


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
