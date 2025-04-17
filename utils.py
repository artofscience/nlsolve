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
        self.qp = self.qp()

        # get dimension of free and prescribed degrees of freedom
        self.nf = np.shape(self.ff)[0] if self.ff is not None else None
        self.np = np.shape(self.qp)[0] if self.qp is not None else None

        # squared norm of load external load and prescribed motion
        self.ff2 = np.dot(self.ff, self.ff) if self.nf is not None else None
        self.qp2 = np.dot(self.qp, self.qp) if self.np is not None else None

    def ff(self) -> State:
        """
        Applied external load.

        :return: None
        """
        return None

    def qp(self) -> State:
        """
        Prescribed motion.

        :return: None
        """
        return None

    def load(self, p: Point) -> State:
        load = 1.0 * self.ff
        load -= self.kfp(p) @ self.qp if self.np else 0.0  # adds to rhs if nonzero prescribed dof
        return load

    def gf(self, p: Point) -> State:
        """
        Internal load associated to the free degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: None
        """
        return None

    def gp(self, p: Point) -> State:
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
        return self.gf(p) - p.y * self.ff

    def rp(self, p: Point) -> State:
        """
        Residual associated to the prescribed degrees of freedom.

        :param p: Point containing current state (motion, load)
        :return: residual associated to the prescribed degrees of freedom
        """

        # prescribed residual is defined as the prescribed internal load PLUS the reaction load
        return self.gp(p) - p.fp

    def combine(self, xf, xp, p: Point) -> State:
        """
        Combines (append) the arrays corresponding to free and prescribed degrees of freedom.

        :param xf: function corresponding to free dofs
        :param xp: function corresponding to prescribed dofs
        :param p: state
        :return: combined array
        """
        x = np.array([])
        if self.nf:
            x = np.append(x, xf(p))
        if self.np:
            x = np.append(x, xp(p))
        return x

    def r(self, p: Point) -> State:
        """
        Retrieve residual load at state p

        :param p: state
        :return: residual load
        """
        return self.combine(self.rf, self.rp, p)

    def g(self, p: Point) -> State:
        """
        Retrieve internal load at state p

        :param p: state
        :return: internal load
        """
        return self.combine(self.gf, self.gp, p)

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
        ddqf, ddqp, ddff, ddfp = 0.0, 0.0, 0.0, 0.0

        if self.nf:
            ddqf = u[:, 0] + y * u[:, 1]
            ddff = y * self.ff
        if self.np:
            ddqp = y * self.qp
            ddfp = self.rp(p) + y * self.kpp(p) @ self.qp
            ddfp += self.kpf(p) @ ddqf if self.nf else 0.0

        return Point(ddqf, ddqp, ddff, ddfp, y)


class Point:
    def __init__(self, qf: State = 0.0, qp: State = 0.0, ff: State = 0.0, fp: State = 0.0, y: float = 0.0) -> None:
        """
        Initialize an (equilibrium) point given it's load and corresponding motion in partitioned format.

        :param qf: free / unknown motion
        :param qp: prescribed motion
        :param ff: external / applied load
        :param fp: reaction load
        :param y: load proportionality parameter
        """
        self.qf = qf
        self.qp = qp
        self.ff = ff
        self.fp = fp
        self.y = y

    @staticmethod
    def combine(xf: np.ndarray | float, xp: np.ndarray | float) -> State:
        """
        Combines (append) the arrays corresponding to free and prescribed degrees of freedom.

        :param xf: array corresponding to free dofs
        :param xp: array corresponding to prescribed dofs
        :return: combined array
        """
        x = np.array([])
        if type(xf) is np.ndarray:
            x = np.append(x, xf)
        if type(xp) is np.ndarray:
            x = np.append(x, xp)
        return x

    @property
    def f(self) -> State:
        """
        Retrieve load at state p

        :param p: state
        :return: load
        """
        return self.combine(self.ff, self.fp)

    @property
    def q(self) -> State:
        """
        Retrieve motion at state p

        :param p: state
        :return: motion
        """
        return self.combine(self.qf, self.qp)

    def __iadd__(self, other: Point) -> Point:
        """
        Adds the content of another Point to this Point.

        :param other: another Point object
        :return: sum of Points
        """
        self.qf += other.qf
        self.qp += other.qp
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
        out.qf *= other
        out.qp *= other
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
        out = deepcopy(Point(self.qf, self.qp, self.ff, self.fp, self.y))
        out += other
        return out

    def __sub__(self, other: Point) -> Point:
        """
        Substraction of two points, returing a third Point.

        :param other: another Point object
        :return: a third Point object that is the substraction
        """
        out = deepcopy(Point(self.qf, self.qp, self.ff, self.fp, self.y))
        out -= other
        return out
